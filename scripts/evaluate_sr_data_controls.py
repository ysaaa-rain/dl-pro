"""One fixed evaluation after validation-only recipe selection; no retraining."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from skimage.metrics import structural_similarity
from sklearn.metrics import f1_score
from dlpro.experiment_core import ContactReader,Classifier,SubpixelSR,ShapeTransform,LABELS,configure,centered_indices,save_json
from dlpro.refinement import ContextCountSR,project_counts
from dlpro.superresolution import mean_pool,scores
out=Path('results/sr_data_controls');selection=json.loads((out/'selection.json').read_text());assert selection['selection_only_validation'];selection_hash=hashlib.sha256((out/'selection.json').read_bytes()).hexdigest();configure(42);device=torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu');models={};scale=None
for recipe in ['sparse_matched','dense_matched','dense_multiscale']:
 for seed in [42,43,44]:
  cp=torch.load(out/f'{recipe}_seed{seed}.pt',weights_only=True);assert cp['protocol_sha256']==selection['protocol_sha256'];scale=cp['scale'];model=ContextCountSR(scale).to(device);model.load_state_dict(cp['state_dict']);model.eval();models[f'{recipe}_seed{seed}']=model
legacy=SubpixelSR().to(device);legacy.load_state_dict(torch.load('results/sr_expanded/model_seed42.pt',weights_only=True)['state_dict']);legacy.eval();baseline_methods=['bilinear','bicubic','projected_bicubic','original_cnn','projected_original_cnn'];methods=baseline_methods+list(models);checks=[]

def infer(method,counts):
 low_counts=torch.from_numpy(mean_pool(counts)[None,None]).to(device);low=torch.log1p(low_counts)/scale
 with torch.no_grad():
  if method in ['bilinear','bicubic']:result=F.interpolate(low,scale_factor=4,mode=method,align_corners=False).clamp(0,1)
  elif method=='projected_bicubic':result=torch.log1p(project_counts(F.interpolate(low_counts,scale_factor=4,mode='bicubic',align_corners=False),low_counts))/scale
  elif method in ['original_cnn','projected_original_cnn']:
   result=legacy(low.clamp(0,1)).clamp(0,1)
   if method=='projected_original_cnn':result=torch.log1p(project_counts(torch.expm1(result*scale),low_counts))/scale
  else:result=models[method](low)
  if method not in ['bilinear','bicubic','original_cnn']:
   error=((F.avg_pool2d(torch.expm1(result*scale),4)-low_counts).abs()/low_counts.clamp_min(1e-3)).max();checks.append(float(error.cpu()));assert checks[-1]<1e-5
 return result.cpu().numpy()[0,0]

frame=pd.read_csv('data/processed/sr_expanded/windows.csv');raw=np.load('data/processed/sr_expanded/raw_counts.npy',mmap_mode='r');native=[]
for idx,row in frame[frame.split=='test'].iterrows():
 counts=raw[idx].astype('float32')*float(row.depth_multiplier);target=np.clip(np.log1p(counts)/scale,0,1)
 for method in methods:
  pred=infer(method,counts);native.append(dict(sample_index=int(idx),group=int(row.group),replicate=int(row.replicate),method=method,**scores(target,np.clip(pred,0,1))))
 print('SR fixed evaluation native',idx,flush=True)
pd.DataFrame(native).to_csv(out/'native_metrics.csv',index=False)
wt=pd.read_csv('data/annotations/samples.csv').query('condition=="37C"').sort_values('replicate');readers={int(s.replicate):ContactReader(s.path) for s in wt.itertuples()};library=readers[1].library_size;known=pd.read_csv('results/sr_context/known_test_windows.csv');folds=pd.read_csv('results/classification_cv/out_of_fold_predictions.csv').set_index('structure_id');classifiers={};transforms={}
for fold in range(5):
 model=Classifier();path='results/classification/model_seed42.pt' if fold==0 else f'results/classification_cv/model_fold{fold}.pt';tf='results/classification/transform.npz' if fold==0 else f'results/classification_cv/transform_fold{fold}.npz';model.load_state_dict(torch.load(path,weights_only=True)['state_dict']);classifiers[fold]=model.eval();transforms[fold]=ShapeTransform.load(tf)
shape=[];predictions=[]
for row in known.itertuples():
 fold=int(folds.loc[row.structure_id,'fold']);tf=transforms[fold];start=int(row.window_start);small=(centered_indices(int(row.center))[0]*100-start)//10;paired={m:[] for m in ['original',*methods]}
 for repeat in [1,2]:
  counts=readers[repeat].window(start//10,2560,10,False)*(library/readers[repeat].library_size)
  def features(decoded):return np.stack([mean_pool(decoded[small:small+1280,small:small+1280],10),mean_pool(decoded,20)]).astype('float32')*1e6/library
  reference=tf.transform(features(counts)[None])[0]
  for method in ['original',*methods]:
   decoded=counts if method=='original' else np.expm1(infer(method,counts)*scale);image=tf.transform(features(decoded)[None]);triangle=np.triu_indices(128,2)
   if method!='original':shape.append(dict(structure_id=row.structure_id,type=row.type,group=int(row.center//100000),replicate=repeat,method=method,oe_ssim_100bp=float(structural_similarity(reference[0],image[0,0],data_range=1)),oe_ssim_200bp=float(structural_similarity(reference[1],image[0,1],data_range=1)),oe_correlation_100bp=float(np.corrcoef(reference[0][triangle],image[0,0][triangle])[0,1])))
   with torch.no_grad():paired[method].append(classifiers[fold](torch.from_numpy(image)).softmax(1).numpy()[0])
 for method,values in paired.items():
  probability=np.mean(values,0);record=dict(structure_id=row.structure_id,true_type=row.type,group=int(row.center//100000),fold=fold,method=method,predicted_type=LABELS[probability.argmax()]);record.update({f'p_{label}':float(probability[k]) for k,label in enumerate(LABELS)});predictions.append(record)
 print('SR fixed evaluation known',row.structure_id,flush=True)
pd.DataFrame(shape).to_csv(out/'shape_metrics.csv',index=False);pred=pd.DataFrame(predictions);pred.to_csv(out/'downstream_predictions.csv',index=False);downstream=[]
for method,g in pred.groupby('method'):downstream.append(dict(method=method,n_loci=len(g),accuracy=float(g.true_type.eq(g.predicted_type).mean()),macro_f1=float(f1_score(g.true_type,g.predicted_type,labels=LABELS,average='macro',zero_division=0))))
pd.DataFrame(downstream).to_csv(out/'downstream_summary.csv',index=False);pd.DataFrame(native).groupby(['method','replicate'])[['psnr','ssim']].mean().to_csv(out/'native_summary.csv');pd.DataFrame(shape).groupby('method')[['oe_ssim_100bp','oe_ssim_200bp','oe_correlation_100bp']].mean().to_csv(out/'shape_summary.csv');save_json(out/'evaluation_protocol.json',dict(frozen_selection_sha256=selection_hash,selected_recipe=selection['selected_recipe'],n_native_test=32,n_known_test=49,max_count_relative_error=max(checks),n_count_probes=len(checks),old_classifier='Unchanged existing two-scale OOF classifier, same as previousSRbenchmark',old_checkpoint='results/sr_expanded/model_seed42.pt',all_9_new_models_evaluated=True,limitations=['Same previously exposed spatial test blocks, retrospective comparison','200bp preservation follows by construction','Three training seeds are not independent biological repeats','No test-based recipe replacement or new training']))
print('FIXED_SR_EVALUATION_COMPLETE',flush=True)

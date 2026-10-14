"""Separate the effect of count conservation from learned second-round improvement."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from skimage.metrics import structural_similarity
from dlpro.experiment_core import ContactReader,SubpixelSR,ShapeTransform,Classifier,LABELS,configure,save_json,centered_indices
from dlpro.refinement import project_counts
from dlpro.superresolution import mean_pool,scores
configure(42);out=Path('results/sr_refined');device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
checkpoint=torch.load('results/sr_expanded/model_seed42.pt',weights_only=True);scale=float(checkpoint['scale']);model=SubpixelSR().to(device);model.load_state_dict(checkpoint['state_dict']);model.eval()
samples=pd.read_csv('data/annotations/samples.csv');wt=samples[samples.condition=='37C'].sort_values('replicate');readers={int(s.replicate):ContactReader(s.path) for s in wt.itertuples()};library=readers[1].library_size

def predict(counts):
 low_counts=mean_pool(counts);low=np.clip(np.log1p(low_counts)/scale,0,1).astype('float32')
 with torch.no_grad():old=model(torch.from_numpy(low[None,None]).to(device)).clamp(0,1).cpu()
 corrected=torch.log1p(project_counts(torch.expm1(old*scale),torch.from_numpy(low_counts[None,None])))/scale
 return old.numpy()[0,0],corrected.numpy()[0,0]

raw=np.load('data/processed/sr_expanded/raw_counts.npy',mmap_mode='r');plan=pd.read_csv('data/processed/sr_expanded/windows.csv');native=[]
for idx,row in plan[plan.split.eq('test')].iterrows():
 counts=raw[idx].astype('float32')*row.depth_multiplier;target=np.clip(np.log1p(counts)/scale,0,1)
 for method,pred in zip(['first_round_cnn','projected_first_round_cnn'],predict(counts)):
  native.append(dict(seed=42,method=method,sample_index=idx,replicate=int(row.replicate),**scores(target,pred)))
pd.DataFrame(native).to_csv(out/'projection_ablation_native.csv',index=False)
known=pd.read_csv('results/sr_refined/known_test_windows.csv');folds=pd.read_csv('results/classification_cv/out_of_fold_predictions.csv').set_index('structure_id');metrics=[];predictions=[];models={};transforms={}
for fold in range(5):
 tf='results/classification/transform.npz' if fold==0 else f'results/classification_cv/transform_fold{fold}.npz';path='results/classification/model_seed42.pt' if fold==0 else f'results/classification_cv/model_fold{fold}.pt'
 transforms[fold]=ShapeTransform.load(tf);c=Classifier();c.load_state_dict(torch.load(path,weights_only=True)['state_dict']);c.eval();models[fold]=c
for row in known.itertuples():
 fold=int(folds.loc[row.structure_id].fold);tf=transforms[fold];paired={m:[] for m in ['first_round_cnn','projected_first_round_cnn']}
 for repeat in [1,2]:
  counts=readers[repeat].window(int(row.window_start)//10,n=2560,bin_bp=10,depth_normalize=False)*(library/readers[repeat].library_size);small=(centered_indices(int(row.center))[0]*100-row.window_start)//10
  def features(c):return tf.transform(np.stack([mean_pool(c[small:small+1280,small:small+1280],10),mean_pool(c,20)])[None].astype('float32')*1e6/library)[0]
  reference=features(counts)
  for method,pred in zip(paired,predict(counts)):
   f=features(np.expm1(pred*scale));metrics.append(dict(structure_id=row.structure_id,type=row.type,replicate=repeat,method=method,oe_ssim_100bp=float(structural_similarity(reference[0],f[0],data_range=1)),oe_ssim_200bp=float(structural_similarity(reference[1],f[1],data_range=1))))
   with torch.no_grad():paired[method].append(F.softmax(models[fold](torch.from_numpy(f[None])),1).numpy()[0])
 for method,probs in paired.items():predictions.append(dict(structure_id=row.structure_id,true_type=row.type,method=method,predicted_type=LABELS[np.mean(probs,0).argmax()]))
 print('projection ablation',row.structure_id,flush=True)
pd.DataFrame(metrics).to_csv(out/'projection_ablation_shape.csv',index=False);pd.DataFrame(predictions).to_csv(out/'projection_ablation_predictions.csv',index=False)
save_json(out/'projection_ablation_protocol.json',dict(seed=42,n_test_patches=32,n_test_loci=49,training='None; reuse first-round saved model, inference projection only',input='Original model receives original clipped input; projection restores measured unclipped40bp block mean',interpretation='200bp conservation follows by construction; compare100bp off-grid shapes and native metrics for learned benefit',test_status='Retrospective positions already evaluated in first round'))

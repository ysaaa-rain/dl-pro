"""Train-only native distance prior; validation chooses postprocessing before testing."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from skimage.metrics import structural_similarity
from dlpro.experiment_core import ContactReader,SubpixelSR,ShapeTransform,Classifier,LABELS,configure,save_json,centered_indices
from dlpro.refinement import project_counts
from dlpro.superresolution import mean_pool,scores
configure(42);out=Path('results/sr_distance_prior')
from scipy.ndimage import gaussian_filter1d
out.mkdir(exist_ok=True);device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
checkpoint=torch.load('results/sr_expanded/model_seed42.pt',weights_only=True);scale=float(checkpoint['scale']);model=SubpixelSR().to(device);model.load_state_dict(checkpoint['state_dict']);model.eval()
samples=pd.read_csv('data/annotations/samples.csv');wt=samples[samples.condition=='37C'].sort_values('replicate');readers={int(s.replicate):ContactReader(s.path) for s in wt.itertuples()};library=readers[1].library_size

raw=np.load('data/processed/sr_expanded/raw_counts.npy',mmap_mode='r');plan=pd.read_csv('data/processed/sr_expanded/windows.csv')
train_ids=np.flatnonzero(plan.split.eq('train'));profile=np.asarray([np.mean([np.diagonal(raw[i],offset=d).mean() for i in train_ids]) for d in range(1024)])
profile=np.exp(gaussian_filter1d(np.log(np.maximum(profile,1e-6)),1));np.savez(out/'distance_profile.npz',expected=profile,train_indices=train_ids)
expect=np.load('results/sr_refined/shape_expectation.npz')['expected'];d100=abs(np.arange(102)[:,None]-np.arange(102)[None,:]);valid=d100>=2

def bases(counts):
    low_counts=mean_pool(counts);low=np.clip(np.log1p(low_counts)/scale,0,1).astype('float32')
    with torch.no_grad():old=model(torch.from_numpy(low[None,None]).to(device)).clamp(0,1).cpu()
    bicubic=F.interpolate(torch.from_numpy(low_counts[None,None]),scale_factor=4,mode='bicubic',align_corners=False)
    return {'cnn':torch.expm1(old*scale),'bicubic':bicubic.clamp_min(1e-8)},torch.from_numpy(low_counts[None,None])

def allocate(prior,low,gamma):
    n=prior.shape[-1];distance=abs(np.arange(n)[:,None]-np.arange(n)[None,:]);e=torch.from_numpy(profile[np.minimum(distance,len(profile)-1)].astype('float32'))[None,None]
    ratio=e/F.interpolate(F.avg_pool2d(e,4),scale_factor=4,mode='nearest')
    counts=project_counts(prior*ratio.pow(gamma),low)
    return (torch.log1p(counts)/scale).numpy()[0,0]

validation=[]
for source in ['cnn','bicubic']:
    for gamma in [0,.5,1]:
        native_losses=[];shape_losses=[]
        for idx,row in plan[plan.split.eq('val')].iterrows():
            counts=raw[idx].astype('float32')*row.depth_multiplier;prior,low=bases(counts);pred=allocate(prior[source],low,gamma);target=np.clip(np.log1p(counts)/scale,0,1);native_losses.append(float(((target-np.clip(pred,0,1))**2).mean()))
            a=np.log1p(mean_pool(np.expm1(pred[:1020,:1020]*scale),10)/expect[d100]);b=np.log1p(mean_pool(counts[:1020,:1020],10)/expect[d100]);shape_losses.append(float(((a-b)[valid]**2).mean()))
        validation.append(dict(source=source,gamma=gamma,val_native_mse=float(np.mean(native_losses)),val_100bp_oe_mse=float(np.mean(shape_losses)),objective=float(np.mean(native_losses)+.1*np.mean(shape_losses))))
choice=min(validation,key=lambda v:v['objective']);pd.DataFrame(validation).to_csv(out/'validation_grid.csv',index=False);save_json(out/'selection.json',choice);print('validation choice',choice,flush=True)

def predict(counts):
    prior,low=bases(counts)
    return allocate(prior['cnn'],low,0),allocate(prior[choice['source']],low,choice['gamma'])

raw=np.load('data/processed/sr_expanded/raw_counts.npy',mmap_mode='r');plan=pd.read_csv('data/processed/sr_expanded/windows.csv');native=[]
for idx,row in plan[plan.split.eq('test')].iterrows():
 counts=raw[idx].astype('float32')*row.depth_multiplier;target=np.clip(np.log1p(counts)/scale,0,1)
 for method,pred in zip(['count_consistent_cnn','distance_prior_selected'],predict(counts)):
  native.append(dict(seed=42,method=method,sample_index=idx,replicate=int(row.replicate),**scores(target,pred)))
  d=abs(np.arange(len(target))[:,None]-np.arange(len(target))[None,:]);error=(target-np.clip(pred,0,1))**2;native[-1].update(psnr_near_200bp=float(-10*np.log10(max(error[d<20].mean(),1e-12))),psnr_far_200bp=float(-10*np.log10(max(error[d>=20].mean(),1e-12))))
pd.DataFrame(native).to_csv(out/'native_metrics.csv',index=False)
known=pd.read_csv('results/sr_refined/known_test_windows.csv');folds=pd.read_csv('results/classification_cv/out_of_fold_predictions.csv').set_index('structure_id');metrics=[];predictions=[];models={};transforms={}
for fold in range(5):
 tf='results/classification/transform.npz' if fold==0 else f'results/classification_cv/transform_fold{fold}.npz';path='results/classification/model_seed42.pt' if fold==0 else f'results/classification_cv/model_fold{fold}.pt'
 transforms[fold]=ShapeTransform.load(tf);c=Classifier();c.load_state_dict(torch.load(path,weights_only=True)['state_dict']);c.eval();models[fold]=c
for row in known.itertuples():
 fold=int(folds.loc[row.structure_id].fold);tf=transforms[fold];paired={m:[] for m in ['count_consistent_cnn','distance_prior_selected']}
 for repeat in [1,2]:
  counts=readers[repeat].window(int(row.window_start)//10,n=2560,bin_bp=10,depth_normalize=False)*(library/readers[repeat].library_size);small=(centered_indices(int(row.center))[0]*100-row.window_start)//10
  def features(c):return tf.transform(np.stack([mean_pool(c[small:small+1280,small:small+1280],10),mean_pool(c,20)])[None].astype('float32')*1e6/library)[0]
  reference=features(counts)
  for method,pred in zip(paired,predict(counts)):
   f=features(np.expm1(pred*scale));metrics.append(dict(structure_id=row.structure_id,type=row.type,replicate=repeat,method=method,oe_ssim_100bp=float(structural_similarity(reference[0],f[0],data_range=1)),oe_ssim_200bp=float(structural_similarity(reference[1],f[1],data_range=1))))
   with torch.no_grad():paired[method].append(F.softmax(models[fold](torch.from_numpy(f[None])),1).numpy()[0])
 for method,probs in paired.items():predictions.append(dict(structure_id=row.structure_id,true_type=row.type,method=method,predicted_type=LABELS[np.mean(probs,0).argmax()]))
 print('projection ablation',row.structure_id,flush=True)
pd.DataFrame(metrics).to_csv(out/'shape_metrics.csv',index=False);pd.DataFrame(predictions).to_csv(out/'classification_predictions.csv',index=False)
save_json(out/'protocol.json',dict(seed=42,n_test_patches=32,n_test_loci=49,training='Native distance expectation from64training matrices only; no new CNN training; Gaussian smoothing sigma1',input='Original model receives original clipped input; projection restores measured unclipped40bp block mean',validation_choices=validation,selected_recipe=choice,interpretation='Within-block distance allocation plus40bp count conservation;200bp preservation by construction; near/far200bp native metrics isolate diagonal-band gain',test_status='Retrospective positions already evaluated in first round'))

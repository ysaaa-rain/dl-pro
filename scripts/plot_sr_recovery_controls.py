"""Three fixed first-per-class100bp recovery examples, both repeats; CPU rendering."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dlpro.experiment_core import ContactReader,ShapeTransform,SubpixelSR,centered_indices
from dlpro.refinement import ContextCountSR,project_counts
from dlpro.superresolution import mean_pool
out=Path('results/sr_data_controls/recovery_examples');out.mkdir(exist_ok=True);torch.set_num_threads(2);known=pd.read_csv('results/sr_context/known_test_windows.csv').groupby('type',sort=False).head(1);folds=pd.read_csv('results/classification_cv/out_of_fold_predictions.csv').set_index('structure_id');wt=pd.read_csv('data/annotations/samples.csv').query('condition=="37C"').sort_values('replicate');readers={int(r.replicate):ContactReader(r.path) for r in wt.itertuples()};library=readers[1].library_size;cp=torch.load('results/sr_data_controls/dense_multiscale_seed42.pt',weights_only=True);scale=cp['scale'];new=ContextCountSR(scale).eval();new.load_state_dict(cp['state_dict']);old=SubpixelSR().eval();old.load_state_dict(torch.load('results/sr_expanded/model_seed42.pt',weights_only=True)['state_dict'])
for row in known.itertuples():
 fold=int(folds.loc[row.structure_id,'fold']);tf=ShapeTransform.load('results/classification/transform.npz' if fold==0 else f'results/classification_cv/transform_fold{fold}.npz');distance=abs(np.arange(128)[:,None]-np.arange(128)[None,:]);fig,axes=plt.subplots(2,5,figsize=(16,6.8),constrained_layout=True);saved={}
 for repeat in [1,2]:
  counts=readers[repeat].window(int(row.window_start)//10,2560,10,False)*(library/readers[repeat].library_size);low=torch.from_numpy(mean_pool(counts)[None,None]);small=(centered_indices(int(row.center))[0]*100-int(row.window_start))//10
  with torch.no_grad():
   lowlog=torch.log1p(low)/scale;prior=F.interpolate(low,scale_factor=4,mode='bicubic',align_corners=False);base=project_counts(prior,low).numpy()[0,0];legacy=old(lowlog.clamp(0,1)).clamp(0,1);projected=project_counts(torch.expm1(legacy*scale),low).numpy()[0,0];pred=torch.expm1(new(lowlog)*scale).numpy()[0,0]
  def view(a):
   coarse=mean_pool(a[small:small+1280,small:small+1280],10)*1e6/library;return np.clip(np.log1p(coarse/tf.expected[0,distance])/tf.scale[0],0,1)
  images=[view(a) for a in [counts,base,projected,pred]]
  for a in images:a[distance<2]=0
  error=abs(images[3]-images[0])-abs(images[1]-images[0]);saved[f'rep{repeat}']=np.stack([*images,error]);titles=['Original','Projected bicubic','Projected originalCNN','Multiscale CNN','Abs error: CNN minus bicubic']
  for col,image in enumerate([*images,error]):
   ax=axes[repeat-1,col];im=ax.imshow(image,origin='lower',cmap='coolwarm' if col==4 else 'magma',vmin=-.1 if col==4 else 0,vmax=.1 if col==4 else 1);ax.set_title(titles[col],fontsize=9);ax.set_xlabel('100bp bin');
   if col==0:ax.set_ylabel(f'WT rep{repeat};100bp bin')
  fig.colorbar(im,ax=axes[repeat-1,-1],shrink=.75,label='Negative = lower error')
 fig.suptitle(f'{row.structure_id}; first held-out example of {row.type}; fixed seed42; not selected by success');fig.savefig(out/f'{row.type}.png',dpi=140);plt.close(fig);np.savez_compressed(out/f'{row.type}.npz',**saved);print('Recovery figure',row.structure_id,flush=True)

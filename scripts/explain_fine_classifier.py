"""OOF multiview gradients and equal-area zero/neutral occlusion diagnostics."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dlpro.experiment_core import ShapeTransform,LABELS,configure,save_json
from dlpro.classification_models import FineScaleClassifier
configure(42);out=Path('results/classification_fine/explanations');out.mkdir(exist_ok=True);f=pd.read_csv('data/processed/known_windows.csv');folds=pd.read_csv('results/classification_cv/out_of_fold_predictions.csv').set_index('structure_id').loc[f.structure_id,'fold'].to_numpy();n=len(f);raw=np.concatenate([np.concatenate([np.load(f'data/processed/fine_rep{r}.npz')['raw'],np.load(f'data/processed/known_rep{r}.npz')['raw']],1) for r in [1,2]]);y=f.type.map(dict(zip(LABELS,range(3)))).to_numpy();rows=[];pictures=set();diagonal=abs(np.arange(128)[:,None]-np.arange(128)[None,:])<2
for fold in range(5):
 tf=ShapeTransform.load(f'results/classification_fine/transform_fold{fold}.npz');x=tf.transform(raw);ids=np.flatnonzero(folds==fold);view_ids=np.concatenate([ids,ids+n]);tx=torch.from_numpy(x[view_ids]);models=[]
 for seed in [42,43,44]:
  model=FineScaleClassifier();model.load_state_dict(torch.load(f'results/classification_fine/fine_consistent_seed{seed}_fold{fold}.pt',weights_only=True)['state_dict']);model.eval();models.append(model)
 def probability(images):
  with torch.no_grad():return np.mean([np.concatenate([F.softmax(m(z),1).numpy() for z in images.split(64)]) for m in models],0)
 original=probability(tx);paired=(original[:len(ids)]+original[len(ids):])/2
 for fill in ['zero','neutral']:
  variants=[]
  for region in ['center','corners']:
   image=tx.clone();value=torch.zeros(3) if fill=='zero' else torch.from_numpy(np.clip(np.log(2)/tf.scale,0,1).astype('float32'));value=value[None,:,None,None]
   if region=='center':image[:,:,48:80,48:80]=value
   else:
    for a,b in [(0,0),(0,112),(112,0),(112,112)]:image[:,:,a:a+16,b:b+16]=value
   image[:,:,torch.from_numpy(diagonal)]=0;prob=probability(image);variants.append((prob[:len(ids)]+prob[len(ids):])/2)
  for local,i in enumerate(ids):rows.append(dict(structure_id=f.structure_id.iloc[i],true_type=f.type.iloc[i],fold=fold,fill=fill,predicted_type=LABELS[paired[local].argmax()],original_true_probability=float(paired[local,y[i]]),center_probability_drop=float(paired[local,y[i]]-variants[0][local,y[i]]),corner_probability_drop=float(paired[local,y[i]]-variants[1][local,y[i]])))
 for label in LABELS:
  if label in pictures or not (f.type.iloc[ids]==label).any():continue
  i=int(ids[np.flatnonzero((f.type.iloc[ids]==label).to_numpy())[0]]);pictures.add(label);input=torch.from_numpy(x[i:i+1]).requires_grad_();target=LABELS.index(label);logits=torch.stack([m(input)[0] for m in models]).mean(0);logits[target].backward();gradient=input.grad.detach().abs().numpy()[0];relative=gradient/max(float(gradient.max()),1e-12);fig,axes=plt.subplots(2,3,figsize=(10,6),constrained_layout=True)
  for channel,bp in enumerate([50,100,200]):
   axes[0,channel].imshow(x[i,channel],origin='lower',vmin=0,vmax=1,cmap='magma');im=axes[1,channel].imshow(relative[channel],origin='lower',vmin=0,vmax=1,cmap='inferno');axes[0,channel].set_title(f'{bp} bp; {128*bp/1000:g}kb view');axes[1,channel].set_title('Relative true-class gradient')
  fig.suptitle(f'{f.structure_id.iloc[i]}; true {label}; rep1; OOF fold{fold}');fig.colorbar(im,ax=axes[1],shrink=.7);fig.savefig(out/f'gradient_{label}.png',dpi=130);plt.close(fig);np.savez_compressed(out/f'gradient_{label}.npz',input=x[i],absolute_gradient=gradient)
 print('explanation fold',fold,flush=True)
frame=pd.DataFrame(rows);frame.to_csv(out/'occlusion.csv',index=False);frame.groupby(['fill','true_type'])[['center_probability_drop','corner_probability_drop']].agg(['mean','median']).to_csv(out/'occlusion_summary.csv');save_json(out/'protocol.json',dict(model='Three-seed fine_consistent ensemble; per-locus OOF models',n_loci=344,gradient='Mean true-class logit gradient for rep1; magnitude normalized jointly across3views per example',example_choice='First encountered held-out example of each type; no success filtering',occlusion='Center32x32 versus four16x16corners, both zero and training-OE1 neutral fill; near diagonals re-masked; repeat probabilities averaged',limitations=['Sensitivity diagnostics, not causal proof or calibrated attention','Zero occlusion creates atypical holes; neutral counterpart retained','Fixed equal-pixel masks cover different physical lengths in the three views','True-class gradient is shown even if the model predicts a different class']))

"""Fixed train/validation experiment; deliberately never computes test metrics."""
import hashlib,json,math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from dlpro.experiment_core import configure,save_json
from dlpro.refinement import ContextCountSR
from dlpro.structure_losses import fit_expectations,structure_terms
from dlpro.superresolution import mean_pool
out=Path('results/sr_data_controls');out.mkdir(exist_ok=True);plan=json.loads(Path('results/performance_protocol/plan.json').read_text())['sr'];protocol_hash=hashlib.sha256(Path('results/performance_protocol/plan.json').read_bytes()).hexdigest();dense=pd.read_csv('data/processed/sr_dense/windows.csv');raw=np.load('data/processed/sr_dense/raw_counts.npy',mmap_mode='r');old=pd.read_csv('data/processed/sr_expanded/windows.csv');original=np.load('data/processed/sr_expanded/raw_counts.npy',mmap_mode='r');scale=json.loads(Path('results/sr_context/protocol.json').read_text())['normalization_scale'];expect=fit_expectations(raw,np.flatnonzero(dense.original_sparse));np.savez_compressed(out/'expectations.npz',**{str(k):v for k,v in expect.items()});device=torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu');configure(42)
for i,row in dense[dense.original_sparse].iterrows():
 match=old.index[(old.start==row.start)&(old.replicate==1)&(old.split=='train')];assert len(match)==1 and np.array_equal(raw[i],original[match[0]])
validation=np.stack([original[i].astype('float32') for i in old.index[old.split=='val']]);assert len(validation)==12

def batch(counts):
 high=torch.from_numpy(counts[:,None]).to(device);low=F.avg_pool2d(high,4);return torch.log1p(low)/scale,high

def objectives(model,low,high,all_scales=True):
 prediction=model(low);native=F.mse_loss(prediction.clamp(0,1),(torch.log1p(high)/scale).clamp(0,1));decoded=torch.expm1(prediction*scale);terms=structure_terms(decoded,high,expect if all_scales else {100:expect[100]});mse=torch.stack([v[0] for v in terms.values()]).mean();ssim=torch.stack([v[1] for v in terms.values()]).mean();return native,mse,ssim

rows=[]
for recipe in plan['recipes']:
 for seed in plan['seeds']:
  checkpoint=out/f'{recipe}_seed{seed}.pt';history_path=out/f'{recipe}_seed{seed}_history.csv'
  if checkpoint.exists() and history_path.exists() and len(pd.read_csv(history_path))==12:
   cp=torch.load(checkpoint,weights_only=True);assert cp['protocol_sha256']==protocol_hash;rows.append(dict(recipe=recipe,seed=seed,selected_step=cp['selected_step'],validation_objective=cp['validation_objective']));print('reuse',recipe,seed,flush=True);continue
  configure(seed);rng=np.random.default_rng(seed);model=ContextCountSR(scale).to(device);optimizer=torch.optim.Adam(model.parameters(),lr=.001);pool=np.flatnonzero(dense.original_sparse) if recipe=='sparse_matched' else np.arange(len(dense));best=float('inf');history=[];losses=[]
  for step in range(1,961):
   model.train();ids=rng.choice(pool,4,replace=False);counts=np.asarray(raw[ids],dtype='float32').copy();reverse=rng.random(4)<.5;counts[reverse]=counts[reverse,::-1,::-1];low,high=batch(counts);optimizer.param_groups[0]['lr']=.0001+.0009*.5*(1+math.cos(math.pi*(step-1)/960));optimizer.zero_grad();native,mse,ssim=objectives(model,low,high,recipe=='dense_multiscale');loss=native+.1*mse+(.05*ssim if recipe=='dense_multiscale' else 0)
   if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
   loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step();losses.append(float(loss.detach().cpu()))
   if step%80==0:
    model.eval();values=[]
    with torch.no_grad():
     for v in np.array_split(validation,3):
      low,high=batch(v);native,mse,ssim=objectives(model,low,high);values.append([float(t.cpu()) for t in [native,mse,ssim]])
    means=np.mean(values,0);common=float(means@[1,.1,.05]);history.append(dict(step=step,training_objective=float(np.mean(losses)),val_native_mse=means[0],val_multiscale_oe_mse=means[1],val_multiscale_ssim_loss=means[2],val_common_objective=common));losses=[]
    if common<best:
     best=common;selected=step;torch.save(dict(state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},scale=scale,seed=seed,recipe=recipe,architecture='ContextCountSR',selected_step=selected,validation_objective=best,protocol_sha256=protocol_hash),checkpoint)
    pd.DataFrame(history).to_csv(history_path,index=False);print(recipe,seed,'step',step,'val common',common,flush=True)
  rows.append(dict(recipe=recipe,seed=seed,selected_step=selected,validation_objective=best))
summary=pd.DataFrame(rows);summary.to_csv(out/'validation_summary.csv',index=False);rank=summary.groupby('recipe').validation_objective.mean();tie={'dense_matched':0,'sparse_matched':1,'dense_multiscale':2};chosen=min(rank.index,key=lambda r:(rank[r],tie[r]));save_json(out/'selection.json',dict(selected_recipe=chosen,validation_objectives=rank.to_dict(),selection_only_validation=True,protocol_sha256=protocol_hash,seeds=plan['seeds'],test_evaluated=False));save_json(out/'training_protocol.json',dict(**plan,device=str(device),scale=scale,n_dense_train=len(dense),n_sparse_train=64,gradient_norm_clip=1,ssim_log_oe_dynamic_range=4,expectation_fit='Original64 training windows only',protocol_sha256=protocol_hash));print('FROZEN_SELECTION',chosen,rank.to_dict(),flush=True)

"""Purged cross-repeat pretraining and matched fine-tuning; validation-only decisions."""
import copy,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import f1_score
from dlpro.experiment_core import LABELS,LENGTH,ShapeTransform,configure,centered_indices,save_json
from dlpro.classification_models import FineScaleClassifier,reverse_genome,repeat_js_divergence
from dlpro.self_supervision import paired_info_nce
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');out=Path('results/classification_repeat_ssl');out.mkdir(exist_ok=True);protocol_hash=hashlib.sha256(Path('results/performance_protocol/plan.json').read_bytes()).hexdigest();f=pd.read_csv('data/processed/known_windows.csv');folds=pd.read_csv('results/classification_cv/out_of_fold_predictions.csv').set_index('structure_id').loc[f.structure_id,'fold'].to_numpy();n=len(f);labels=f.type.map(dict(zip(LABELS,range(3)))).to_numpy();raw=np.concatenate([np.concatenate([np.load(f'data/processed/fine_rep{r}.npz')['raw'],np.load(f'data/processed/known_rep{r}.npz')['raw']],1) for r in [1,2]]);scan=pd.read_csv('data/processed/scan_windows.csv');scanraw=[np.concatenate([np.load(f'data/processed/fine_scan_rep{r}.npz')['raw'],np.load(f'data/processed/scan_rep{r}.npz')['raw']],1) for r in [1,2]];starts=np.asarray([centered_indices(c)[1]*200 for c in scan.center]);ends=starts+25600;decisions=[];valrecords=[];poolrecords=[]

def metric(prob,truth):
    f1=f1_score(truth,prob.argmax(1),labels=[0,1,2],average=None,zero_division=0);nll=-np.log(np.maximum(prob[np.arange(len(truth)),truth],1e-8)).mean();return f1,float(nll)

def probability(model,images):
    model.eval()
    with torch.no_grad():return np.concatenate([model(b).softmax(1).cpu().numpy() for b in images.split(64)])

for fold in range(5):
 test=folds==fold;val=folds==(fold+1)%5;train=~(test|val);held=f[test|val];overlap=(starts[:,None]<held.window_end.to_numpy()[None])&(ends[:,None]>held.window_start.to_numpy()[None]);eligible=~overlap.any(1)&(starts>=0)&(ends<=LENGTH);pool=[]
 for i in np.flatnonzero(eligible):
  if all(abs(int(scan.center.iloc[i])-int(scan.center.iloc[j]))>=25600 for j in pool):pool.append(i)
 assert len(pool)>=24;assert not overlap[pool].any();tf=ShapeTransform.load(f'results/classification_fine/transform_fold{fold}.npz');tf.save(out/f'transform_fold{fold}.npz');tx=torch.from_numpy(tf.transform(raw)).to(device);sx=[torch.from_numpy(tf.transform(a[pool])).to(device) for a in scanraw];val_ids=np.flatnonzero(val);valviews=tx[np.concatenate([val_ids,val_ids+n])];pools=[np.flatnonzero(train&(labels==c)) for c in range(3)];vprob={}
 for i in pool:poolrecords.append(dict(fold=fold,scan_index=int(i),start=int(starts[i]),end=int(ends[i]),held_overlap=False))
 for recipe in ['matched_control','repeat_pretrained']:
  seedprob=[]
  for seed in [42,43,44]:
   configure(seed);model=FineScaleClassifier().to(device);checkpoint=out/f'{recipe}_seed{seed}_fold{fold}.pt';history_path=out/f'{recipe}_seed{seed}_fold{fold}_history.csv'
   if checkpoint.exists() and history_path.exists() and len(pd.read_csv(history_path))==35:
    cp=torch.load(checkpoint,weights_only=True);assert cp['protocol_sha256']==protocol_hash;model.load_state_dict(cp['state_dict']);pp=probability(model,valviews);seedprob.append((pp[:len(val_ids)]+pp[len(val_ids):])/2);print('reuse',recipe,fold,seed,flush=True);continue
   if recipe=='repeat_pretrained':
    projection=nn.Sequential(nn.Linear(96,96),nn.ReLU(),nn.Linear(96,64)).to(device);optimizer=torch.optim.AdamW(list(model.parameters())+list(projection.parameters()),lr=.001,weight_decay=.001);rng=np.random.default_rng(seed);prehistory=[]
    for epoch in range(30):
     model.train();losses=[]
     for _ in range(int(np.ceil(len(pool)/24))):
      ids=rng.choice(len(pool),24,replace=False);reverse=torch.from_numpy(rng.random(24)<.5).to(device);views=[]
      for source in sx:
       image=reverse_genome(source[ids],reverse);amplitude=torch.from_numpy(rng.uniform(.8,1.2,(24,1,1,1)).astype('float32')).to(device);views.append((image*amplitude).clamp(0,1))
      optimizer.zero_grad();loss=paired_info_nce(projection(model.encode(views[0])),projection(model.encode(views[1])));loss.backward();optimizer.step();losses.append(float(loss.detach()))
     prehistory.append(dict(epoch=epoch+1,train_info_nce=float(np.mean(losses))))
    pd.DataFrame(prehistory).to_csv(out/f'pretrain_seed{seed}_fold{fold}_history.csv',index=False);torch.save(dict(state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},fold=fold,seed=seed,n_unlabelled_locations=len(pool),protocol_sha256=protocol_hash),out/f'pretrained_seed{seed}_fold{fold}.pt')
   # Same fine-tuning RNG, optimizer, batches and epochs in both arms.
   configure(seed);rng=np.random.default_rng(seed);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001);best=(-1,float('-inf'));history=[]
   for epoch in range(35):
    model.train();losses=[]
    for _ in range(int(np.ceil(2*train.sum()/30))):
     locations=np.concatenate([rng.choice(p,5,replace=True) for p in pools]);rng.shuffle(locations);ids=np.concatenate([locations,locations+n]);reverse=rng.random(len(locations))<.5;image=reverse_genome(tx[ids],torch.from_numpy(np.tile(reverse,2)).to(device));truth=torch.from_numpy(np.tile(labels[locations],2)).long().to(device);optimizer.zero_grad();logits=model(image);loss=F.cross_entropy(logits,truth,label_smoothing=.05)+.2*repeat_js_divergence(logits);loss.backward();optimizer.step();losses.append(float(loss.detach()))
    pp=probability(model,valviews);pp=(pp[:len(val_ids)]+pp[len(val_ids):])/2;f1,nll=metric(pp,labels[val]);score=(float(f1.mean()),-nll);history.append(dict(epoch=epoch+1,training_objective=float(np.mean(losses)),val_macro_f1=score[0],val_chid_f1=float(f1[0]),val_nll=nll))
    if score>best:best=score;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};selected=epoch+1
   model.load_state_dict(state);model.eval();torch.save(dict(state_dict=state,fold=fold,seed=seed,recipe=recipe,selected_epoch=selected,protocol_sha256=protocol_hash),checkpoint);pd.DataFrame(history).to_csv(history_path,index=False);pp=probability(model,valviews);seedprob.append((pp[:len(val_ids)]+pp[len(val_ids):])/2);print(recipe,'fold',fold,'seed',seed,'val',best[0],flush=True)
  mean=np.mean(seedprob,0);grid=[]
  for factor in [.5,.75,1,1.25]:
   calibrated=mean*np.array([factor,1,1]);calibrated/=calibrated.sum(1,keepdims=True);f1,nll=metric(calibrated,labels[val]);grid.append((float(f1.mean()),float(f1[0]),-nll,factor,calibrated,f1))
  chosen=max(grid,key=lambda a:(a[0],a[1],a[2]));vprob[recipe]=chosen
  for local,i in enumerate(val_ids):
   row=dict(fold=fold,structure_id=f.structure_id.iloc[i],recipe=recipe,true_type=f.type.iloc[i],chid_factor=chosen[3]);row.update({f'p_{label}':float(chosen[4][local,k]) for k,label in enumerate(LABELS)});valrecords.append(row)
 control=vprob['matched_control'];ssl=vprob['repeat_pretrained'];accept=ssl[0]>=control[0] and ssl[1]>control[1] and np.all(ssl[5][1:]>=control[5][1:]-.03)
 decisions.append(dict(fold=fold,selected_recipe='repeat_pretrained' if accept else 'matched_control',n_unlabelled_train=len(pool),control_chid_factor=control[3],ssl_chid_factor=ssl[3],control_val_f1=control[5].tolist(),ssl_val_f1=ssl[5].tolist(),selection_only_validation=True));print('FROZEN_FOLD_SELECTION',decisions[-1],flush=True)
 pd.DataFrame(valrecords).to_csv(out/'validation_predictions.csv',index=False);pd.DataFrame(poolrecords).to_csv(out/'pretraining_pool.csv',index=False)
save_json(out/'decisions.json',dict(protocol_sha256=protocol_hash,folds=decisions,test_evaluated=False,device=str(device)));print('ALL_CLASSIFICATION_DECISIONS_FROZEN',flush=True)

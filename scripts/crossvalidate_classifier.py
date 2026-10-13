"""Five grouped folds, fixed training recipe; every locus predicted out of fold."""
import copy
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,f1_score
from dlpro.experiment_core import LABELS,ShapeTransform,Classifier,configure,save_json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out=Path('results/classification_cv');out.mkdir(parents=True,exist_ok=True)
f=pd.read_csv('data/processed/known_windows.csv');n=len(f)
raw=np.concatenate([np.load(f'data/processed/known_rep{r}.npz')['raw'] for r in [1,2]])
y0=f.type.map(dict(zip(LABELS,range(3)))).to_numpy();y=np.tile(y0,2);folds=np.zeros(n,int)
for fold,(_,idx) in enumerate(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(f,f.type,f.group)):folds[idx]=fold
assert np.array_equal(folds==0,f.split.eq('test').to_numpy())
assert f.assign(fold=folds).groupby('group').fold.nunique().max()==1
prediction=np.zeros((n,3),float);summaries=[]
for fold in range(5):
    test=folds==fold;val=folds==(fold+1)%5;train0=~(test|val);train=np.tile(train0,2)
    if fold==0:
        original=pd.read_csv('results/classification/predictions.csv');original=original[original.seed==42].set_index('structure_id')
        pp=original.loc[f.loc[test,'structure_id'],[f'p_{label}' for label in LABELS]].to_numpy()
        prediction[test]=pp;selected=int(pd.read_csv('results/classification/metrics.csv').query('seed==42').selected_epoch.iloc[0])
    else:
        configure(42);transform=ShapeTransform().fit(raw[train]);transform.save(out/f'transform_fold{fold}.npz');tx=torch.from_numpy(transform.transform(raw));ty=torch.from_numpy(y).long()
        counts=np.bincount(y[train],minlength=3);weight=torch.from_numpy((train.sum()/(3*counts)).astype(np.float32))
        model=Classifier();optim=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001);rng=np.random.default_rng(42);best=(-1,float('-inf'));state=None;history=[]
        for epoch in range(35):
            model.train();losses=[]
            for ids in np.array_split(rng.permutation(np.flatnonzero(train)),int(np.ceil(train.sum()/32))):
                optim.zero_grad();loss=F.cross_entropy(model(tx[ids]),ty[ids],weight=weight);loss.backward();optim.step();losses.append(float(loss.detach()))
            model.eval()
            with torch.no_grad():prob=np.concatenate([F.softmax(model(z),1).numpy() for z in tx.split(64)])
            paired=(prob[:n]+prob[n:])/2;vp=paired[val];vy=y0[val]
            score=float(f1_score(vy,vp.argmax(1),labels=[0,1,2],average='macro',zero_division=0));nll=float(-np.log(np.maximum(vp[np.arange(len(vy)),vy],1e-8)).mean())
            history.append(dict(epoch=epoch+1,train_loss=float(np.mean(losses)),val_macro_f1=score,val_nll=nll))
            if (score,-nll)>best:best=(score,-nll);state=copy.deepcopy(model.state_dict());selected=epoch+1
        model.load_state_dict(state);model.eval()
        with torch.no_grad():prob=np.concatenate([F.softmax(model(z),1).numpy() for z in tx.split(64)])
        pp=((prob[:n]+prob[n:])/2)[test];prediction[test]=pp
        torch.save(dict(state_dict=state,seed=42,fold=fold,selected_epoch=selected,architecture='Classifier'),out/f'model_fold{fold}.pt')
        pd.DataFrame(history).to_csv(out/f'history_fold{fold}.csv',index=False)
    record=dict(fold=fold,n_test=int(test.sum()),n_groups=int(f.loc[test,'group'].nunique()),selected_epoch=selected,accuracy=float(accuracy_score(y0[test],pp.argmax(1))),macro_f1=float(f1_score(y0[test],pp.argmax(1),labels=[0,1,2],average='macro',zero_division=0)))
    summaries.append(record);print(record,flush=True)
pred=prediction.argmax(1);frame=f[['structure_id','type','group']].copy();frame['fold']=folds;frame['prediction']=[LABELS[p] for p in pred]
for c,label in enumerate(LABELS):frame[f'p_{label}']=prediction[:,c]
frame.to_csv(out/'out_of_fold_predictions.csv',index=False);pd.DataFrame(summaries).to_csv(out/'fold_metrics.csv',index=False)
cm=confusion_matrix(y0,pred,labels=[0,1,2]);pd.DataFrame(cm,index=LABELS,columns=LABELS).to_csv(out/'confusion.csv')
save_json(out/'classification_report.json',classification_report(y0,pred,labels=[0,1,2],target_names=LABELS,output_dict=True,zero_division=0))
# Resample whole overlap groups, retaining neighboring structures together.
rng=np.random.default_rng(123);groups=f.group.unique();boot=[]
for _ in range(1000):
    chosen=rng.choice(groups,len(groups),replace=True);idx=np.concatenate([np.flatnonzero(f.group.to_numpy()==g) for g in chosen])
    boot.append([accuracy_score(y0[idx],pred[idx]),f1_score(y0[idx],pred[idx],labels=[0,1,2],average='macro',zero_division=0)])
limits=np.quantile(boot,[.025,.975],axis=0)
summary=dict(n_loci=n,n_groups=len(groups),accuracy=float(accuracy_score(y0,pred)),macro_f1=float(f1_score(y0,pred,labels=[0,1,2],average='macro',zero_division=0)),group_bootstrap_95_ci=dict(accuracy=limits[:,0].tolist(),macro_f1=limits[:,1].tolist()),split_seed=42,training_seed=42,folds=5,epochs=35,
    validation_rule='Next cyclic fold; other three folds train; held-out fold never selects epoch',normalization='Fit independently on training loci of each fold',fold0_reused='Original seed42 held-out predictions, not retrained',
    limitations=['same-study cross-validation, not independent source validation','bootstrap intervals reflect overlap-group resampling, not biological replicate sampling','labels are manually annotated and class boundaries may overlap'])
save_json(out/'summary.json',summary)
fig,ax=plt.subplots(figsize=(5,4));im=ax.imshow(cm,cmap='Blues');ax.set_xticks(range(3),LABELS);ax.set_yticks(range(3),LABELS);ax.set(xlabel='Predicted',ylabel='True',title='Grouped 5-fold out-of-fold predictions')
for i in range(3):
    for j in range(3):ax.text(j,i,str(cm[i,j]),ha='center',va='center')
fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(out/'confusion.png',dpi=150);plt.close(fig)
print(summary,flush=True)

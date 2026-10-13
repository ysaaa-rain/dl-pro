"""Frozen spatial holdout, class imbalance controls, three seeds and explanation."""
import copy,sys,warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix,classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dlpro.experiment_core import LABELS,ShapeTransform,Classifier,configure,save_json

out=Path('results/classification');out.mkdir(parents=True,exist_ok=True)
f=pd.read_csv('data/processed/known_windows.csv');n=len(f)
raw=np.concatenate([np.load(f'data/processed/known_rep{r}.npz')['raw'] for r in [1,2]])
y=np.tile(f.type.map(dict(zip(LABELS,range(3)))).to_numpy(),2)
train=np.tile(f.split.eq('train').to_numpy(),2);val=f.split.eq('val').to_numpy();test=f.split.eq('test').to_numpy()
transform=ShapeTransform().fit(raw[train]);transform.save(out/'transform.npz')
x=transform.transform(raw);tx=torch.from_numpy(x);ty=torch.from_numpy(y).long()


def metric(truth,pred):
    return dict(accuracy=float(accuracy_score(truth,pred)),macro_f1=float(f1_score(truth,pred,labels=[0,1,2],average='macro',zero_division=0)),balanced_accuracy=float(np.mean([np.mean(pred[truth==c]==c) for c in range(3)])))


def paired(prob):return (prob[:n]+prob[n:])/2


features=np.concatenate([np.stack([np.log1p(raw[:,c].mean(axis=(1,2))),np.log1p(raw[:,c].std(axis=(1,2))),np.log1p(np.quantile(raw[:,c],.9,axis=(1,2)))],axis=1) for c in [0,1]],axis=1)
baselines={'density_only':features,'annotation_length_only':np.tile(np.log1p(f.length_bp.to_numpy()),2)[:,None],'shape_PCA_logreg':x.reshape(len(x),-1)}
baseline_audits=[]
rows=[];truth=f.type.map(dict(zip(LABELS,range(3)))).to_numpy()[test]
majority=int(np.argmax(np.bincount(y[train],minlength=3)))
rows.append(dict(method='majority',**metric(truth,np.full(test.sum(),majority))))
for name,values in baselines.items():
    values=values.astype(np.float64)
    if name=='shape_PCA_logreg':model=make_pipeline(PCA(n_components=16,svd_solver='randomized',random_state=42),LogisticRegression(class_weight='balanced',max_iter=3000,random_state=42))
    else:model=make_pipeline(StandardScaler(),LogisticRegression(class_weight='balanced',max_iter=3000,random_state=42))
    if not np.isfinite(values).all():raise ValueError('Nonfinite baseline features')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model.fit(values[train],y[train]);sample_prob=model.predict_proba(values)
    # Independently calculate the linear predictor with einsum, without BLAS.
    pre,lr=model.steps[0][1],model.steps[-1][1]
    if name=='shape_PCA_logreg':
        zz=np.einsum('nd,kd->nk',values[:5].astype(float)-pre.mean_,pre.components_,optimize=False)
    else:zz=(values[:5]-pre.mean_)/pre.scale_
    logits=np.einsum('nd,kd->nk',zz,lr.coef_,optimize=False)+lr.intercept_
    reference=np.exp(logits-logits.max(1,keepdims=True));reference/=reference.sum(1,keepdims=True)
    delta=float(np.max(abs(reference-sample_prob[:5])))
    if not np.isfinite(sample_prob).all() or delta>1e-6:raise ValueError(f'Baseline {name} failed independent arithmetic check: {delta}')
    with warnings.catch_warnings(record=True) as extra:
        warnings.simplefilter('always');ztrain=pre.transform(values[train])
    logits_train=np.einsum('nd,kd->nk',ztrain,lr.coef_,optimize=False)+lr.intercept_
    ptr=np.exp(logits_train-logits_train.max(1,keepdims=True));ptr/=ptr.sum(1,keepdims=True)
    truth_onehot=np.eye(3)[y[train]];cw=len(ztrain)/(3*np.bincount(y[train],minlength=3));sample_weight=cw[y[train]]
    residual=(ptr-truth_onehot)*sample_weight[:,None]
    gradient=np.einsum('ni,nj->ij',residual,ztrain,optimize=False)/sample_weight.sum()+lr.coef_/(lr.C*sample_weight.sum())
    stationarity=float(max(abs(gradient).max(),abs(residual.sum(0)/sample_weight.sum()).max()))
    if not np.isfinite(stationarity) or stationarity>.01:raise ValueError(f'Baseline stationarity failed: {name} {stationarity}')
    baseline_audits.append(dict(method=name,finite_features=True,finite_probabilities=True,independent_probability_max_error=delta,independent_gradient_max_abs=stationarity,blas_warning_count=len(caught)+len(extra),warnings=sorted(set(str(w.message) for w in list(caught)+list(extra)))))
    prob=paired(sample_prob);pred=prob[test].argmax(1)
    rows.append(dict(method=name,**metric(truth,pred)))
    pd.DataFrame(confusion_matrix(truth,pred,labels=[0,1,2]),index=LABELS,columns=LABELS).to_csv(out/f'{name}_confusion.csv')
pd.DataFrame(rows).to_csv(out/'baselines.csv',index=False)
save_json(out/'baseline_numeric_audit.json',baseline_audits)
if '--baselines-only' in sys.argv:raise SystemExit(0)
results=[];all_predictions=[]
for seed in [42,43,44]:
    configure(seed);model=Classifier();optim=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
    counts=np.bincount(y[train],minlength=3);weights=torch.from_numpy((train.sum()/(3*counts)).astype(np.float32))
    rng=np.random.default_rng(seed);best=(-1,float('-inf'));state=None;history=[]
    for epoch in range(35):
        model.train();losses=[]
        for ids in np.array_split(rng.permutation(np.flatnonzero(train)),int(np.ceil(train.sum()/32))):
            optim.zero_grad();loss=F.cross_entropy(model(tx[ids]),ty[ids],weight=weights);loss.backward();optim.step();losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            prob=np.concatenate([torch.softmax(model(z),1).numpy() for z in tx.split(64)])
        vp=paired(prob)[val];vy=f.type.map(dict(zip(LABELS,range(3)))).to_numpy()[val]
        score=float(f1_score(vy,vp.argmax(1),average='macro',labels=[0,1,2],zero_division=0))
        nll=float(-np.log(np.maximum(vp[np.arange(len(vy)),vy],1e-8)).mean())
        history.append(dict(epoch=epoch+1,train_loss=float(np.mean(losses)),val_macro_f1=score,val_nll=nll))
        if (score,-nll)>best:best=(score,-nll);state=copy.deepcopy(model.state_dict());selected=epoch+1
        if epoch%5==0:print(f'CNN seed {seed} epoch {epoch+1}: val macro-F1 {score:.3f}',flush=True)
    model.load_state_dict(state);model.eval()
    torch.save(dict(state_dict=state,seed=seed,labels=LABELS,selected_epoch=selected,architecture='Classifier'),out/f'model_seed{seed}.pt')
    pd.DataFrame(history).to_csv(out/f'history_seed{seed}.csv',index=False)
    with torch.no_grad():prob=np.concatenate([torch.softmax(model(z),1).numpy() for z in tx.split(64)])
    pp=paired(prob)[test];pred=pp.argmax(1);entry=dict(seed=seed,selected_epoch=selected,val_macro_f1=best[0],**metric(truth,pred));results.append(entry)
    matrix=confusion_matrix(truth,pred,labels=[0,1,2]);pd.DataFrame(matrix,index=LABELS,columns=LABELS).to_csv(out/f'confusion_seed{seed}.csv')
    save_json(out/f'report_seed{seed}.json',classification_report(truth,pred,target_names=LABELS,labels=[0,1,2],output_dict=True,zero_division=0))
    for local,idx in enumerate(np.flatnonzero(test)):
        record=dict(structure_id=f.iloc[idx].structure_id,true_type=f.iloc[idx].type,seed=seed,predicted_type=LABELS[pred[local]])
        record.update({f'p_{label}':float(pp[local,c]) for c,label in enumerate(LABELS)})
        record.update(rep1_prediction=LABELS[prob[idx].argmax()],rep2_prediction=LABELS[prob[idx+n].argmax()]);all_predictions.append(record)
    fig,ax=plt.subplots(figsize=(5,4));im=ax.imshow(matrix,cmap='Blues');ax.set_xticks(range(3),LABELS);ax.set_yticks(range(3),LABELS)
    ax.set(xlabel='Predicted',ylabel='True',title=f'CNN seed {seed}; held-out loci')
    for i in range(3):
        for j in range(3):ax.text(j,i,str(matrix[i,j]),ha='center',va='center')
    fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(out/f'confusion_seed{seed}.png',dpi=150);plt.close(fig)
    print(entry,flush=True)
    if seed==42:
        fig,axes=plt.subplots(3,4,figsize=(12,9),constrained_layout=True)
        for row,label in enumerate(LABELS):
            idx=int(np.flatnonzero(test&(f.type==label).to_numpy())[0]);inp=tx[idx:idx+1].clone().requires_grad_()
            logits=model(inp);logits[0,row].backward();sal=inp.grad.detach().abs().numpy()[0]
            for channel in range(2):
                axes[row,channel*2].imshow(x[idx,channel],cmap='magma',vmin=0,vmax=1,origin='lower')
                axes[row,channel*2+1].imshow(sal[channel]/max(sal[channel].max(),1e-9),cmap='inferno',vmin=0,vmax=1,origin='lower')
                axes[row,channel*2].set_title(f'{label} {f.iloc[idx].structure_id}\n{[100,200][channel]} bp input')
                axes[row,channel*2+1].set_title(f'Gradient; pred {LABELS[logits.argmax(1).item()]}')
        fig.savefig(out/'saliency.png',dpi=130);plt.close(fig)
        ids=np.flatnonzero(test);original=tx[ids].clone();center=original.clone();corners=original.clone()
        center[:,:,48:80,48:80]=0
        for aa,bb in [(0,0),(0,112),(112,0),(112,112)]:corners[:,:,aa:aa+16,bb:bb+16]=0
        with torch.no_grad():
            p0=F.softmax(model(original),1).numpy();pc=F.softmax(model(center),1).numpy();pb=F.softmax(model(corners),1).numpy()
        occlusion=pd.DataFrame(dict(structure_id=f.iloc[ids].structure_id.to_numpy(),true_type=f.iloc[ids].type.to_numpy(),center_probability_drop=p0[np.arange(len(ids)),truth]-pc[np.arange(len(ids)),truth],corner_probability_drop=p0[np.arange(len(ids)),truth]-pb[np.arange(len(ids)),truth]))
        occlusion.to_csv(out/'occlusion.csv',index=False)
        # Cross-repeat metrics are descriptive: both repeats trained on training loci.
        repeat_metrics=[dict(replicate=r+1,**metric(truth,prob[r*n:(r+1)*n][test].argmax(1))) for r in range(2)]
        save_json(out/'repeat_metrics_seed42.json',repeat_metrics)
pd.DataFrame(results).to_csv(out/'metrics.csv',index=False);pd.DataFrame(all_predictions).to_csv(out/'predictions.csv',index=False)
save_json(out/'protocol.json',dict(frozen_split_seed=42,training_seeds=[42,43,44],epochs=35,normalization_fit='training-only, both WT repeats',metric_unit='structure locus; average repeat probabilities',model_choice='validation macro-F1 then validation NLL; no test selection',input='fixed 12.8kb and 25.6kb O/E shapes; no length/type metadata',diagonals_masked=2,baselines=list(baselines),length_baseline='diagnostic shortcut using annotation metadata, not fair deployable image-only baseline',limitations=['same-study spatial holdout; not source-disjoint generalization','three seeds share same split and are not independent biological samples','saliency and occlusion are diagnostic, not proof of biological causality']))

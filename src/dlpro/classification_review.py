"""Frozen-model classification diagnostics, label-symmetric calibration and controlled masks."""
import itertools
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from sklearn.metrics import precision_recall_fscore_support,roc_auc_score,average_precision_score
from dlpro.coverage_support import coarse_support
from dlpro.experiment_core import LABELS,LENGTH

def load_inputs():
    frame=pd.read_csv('data/processed/known_windows.csv');raw=[];support=[]
    for repeat in [1,2]:
        raw.append(np.concatenate([np.load(f'data/processed/fine_rep{repeat}.npz')['raw'],np.load(f'data/processed/known_rep{repeat}.npz')['raw']],1))
        coverage=np.load(f'data/processed/coverage_rep{repeat}.npz')['coverage'];ids=np.arange(len(coverage));width=np.minimum(10,LENGTH-ids*10);bins=ids//5;total=(LENGTH+49)//50
        fraction=np.bincount(bins,weights=width*(coverage>0),minlength=total)/np.bincount(bins,weights=width,minlength=total)
        starts=np.floor((frame.center.to_numpy()+25)/50).astype(int)-64;fine=fraction[(starts[:,None]+np.arange(128))%total]
        support.append(np.concatenate([fine[:,None],coarse_support(coverage,frame.center.to_numpy())],1))
    folds=pd.read_csv('results/classification_cv/out_of_fold_predictions.csv').set_index('structure_id').loc[frame.structure_id,'fold'].to_numpy()
    return frame,folds,np.concatenate(raw),np.concatenate(support)

def metrics(y,pred,k):
    cm=np.bincount(np.asarray(y)*k+np.asarray(pred),minlength=k*k).reshape(k,k)
    return from_confusion(cm)

def from_confusion(cm):
    tp=np.diagonal(cm,axis1=-2,axis2=-1);support=cm.sum(-1);called=cm.sum(-2)
    precision=np.divide(tp,called,out=np.zeros_like(tp,dtype=float),where=called>0);recall=np.divide(tp,support,out=np.zeros_like(tp,dtype=float),where=support>0)
    f1=np.divide(2*tp,support+called,out=np.zeros_like(tp,dtype=float),where=support+called>0)
    return dict(accuracy=tp.sum(-1)/cm.sum(axis=(-2,-1)),macro_f1=f1.mean(-1),precision=precision,recall=recall,f1=f1,support=support)

def paired_metric_intervals(y,a,b,groups,k,seed,draws):
    unique,inv=np.unique(groups,return_inverse=True);cms=[]
    for pred in [a,b]:
        cm=np.zeros((len(unique),k,k));np.add.at(cm,(inv,y,pred),1);cms.append(cm)
    w=np.random.default_rng(seed).multinomial(len(unique),np.full(len(unique),1/len(unique)),draws)
    samples=[from_confusion(np.einsum('bg,gij->bij',w,cm)) for cm in cms];point=[from_confusion(cm.sum(0)) for cm in cms];rows=[]
    for name in ['accuracy','macro_f1','precision','recall','f1']:
        delta=samples[0][name]-samples[1][name];effect=point[0][name]-point[1][name]
        for index in (range(k) if np.ndim(effect) else [None]):
            values=delta[:,index] if index is not None else delta;value=effect[index] if index is not None else effect
            lo,hi=np.quantile(values,[.025,.975]);rows.append(dict(metric=name,class_index=index,delta=float(value),low=float(lo),high=float(hi),groups=len(unique)))
    return rows

def calibrate_bias(prob,bias):
    logp=np.log(np.clip(prob,1e-12,1))+np.asarray(bias);logp-=logp.max(1,keepdims=True);out=np.exp(logp);return out/out.sum(1,keepdims=True)

def choose_bias(prob,truth,grid):
    """Only accepts the inner validation rows, never outer labels or IDs."""
    k=prob.shape[1];seen=set();trials=[];best=None
    for raw in itertools.product(grid,repeat=k):
        bias=np.asarray(raw,float);bias-=bias.mean();key=tuple(np.round(bias,12))
        if key in seen:continue
        seen.add(key);p=calibrate_bias(prob,bias);mf=float(metrics(truth,p.argmax(1),k)['macro_f1']);nll=float(-np.log(p[np.arange(len(truth)),truth].clip(1e-12)).mean());quality=(mf,-float(bias@bias),-nll)
        trials.append(dict(bias=bias.tolist(),macro_f1=mf,nll=nll))
        if best is None or quality>best[0]:best=(quality,bias)
    return best[1],trials

def check_spatial_partition(starts,ends,partition):
    overlap=(starts[:,None]<ends[None,:])&(ends[:,None]>starts[None,:]);bad=overlap&(partition[:,None]!=partition[None,:])
    if bad.any():raise ValueError('Overlapping model contexts cross partition boundaries')
    return True

def matched_masks(target,valid,seed):
    """Match counts at each exact diagonal distance; retain failure if background is too small."""
    target=np.asarray(target,bool)&valid;distance=abs(np.arange(len(target))[:,None]-np.arange(len(target))[None,:]);upper=np.triu(np.ones_like(target),2);chosen=np.zeros_like(target);rng=np.random.default_rng(seed)
    for d in np.unique(distance[target&upper]):
        need=int((target&upper&(distance==d)).sum());allowed=np.flatnonzero(valid&upper&~target&(distance==d))
        if len(allowed)<need:return target,None
        chosen.flat[rng.choice(allowed,need,replace=False)]=True
    chosen|=chosen.T
    return target,chosen

def orientation_features(raw,expected,valid_rows):
    """Reversal-invariant directional texture; no annotation geometry, class or position inputs."""
    n=raw.shape[-1];d=abs(np.arange(n)[:,None]-np.arange(n)[None,:]);features=[]
    for channel in range(raw.shape[1]):
        valid=(d[None]>=2)&valid_rows[:,channel,:,None]&valid_rows[:,channel,None,:]
        oe=raw[:,channel]/expected[channel,d][None];amplitude=(oe*valid).sum((1,2))/valid.sum((1,2)).clip(1);field=np.where(valid,np.log1p(oe/amplitude[:,None,None].clip(1e-10)),0)
        for sigma in [1,2,4]:
            support=gaussian_filter(valid.astype(float),(0,sigma,sigma));smooth=gaussian_filter(field,(0,sigma,sigma))/support.clip(1e-6);gx,gy=np.gradient(smooth,axis=(1,2));xx=np.gradient(gx,axis=1);yy=np.gradient(gy,axis=2);xy=(np.gradient(gx,axis=2)+np.gradient(gy,axis=1))/2
            responses=[-(xx+2*xy+yy)/2,-(xx-2*xy+yy)/2,-(xx+yy)/2,np.sqrt((xx-yy)**2+4*xy**2)]
            for lo,hi in [(2,16),(16,64)]:
                keep=valid&(d[None]>=lo)&(d[None]<hi)&(support>=.9)
                for response in responses:
                    for quantile in [.5,.9]:features.append(np.asarray([np.quantile(v[m]*sigma*sigma,quantile) if m.any() else 0 for v,m in zip(response,keep)]))
    result=np.stack(features,1);assert np.isfinite(result).all();return result

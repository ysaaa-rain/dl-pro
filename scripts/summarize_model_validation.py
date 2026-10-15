"""Paired whole-group confidence intervals across all frozen experimental recipes."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from dlpro.experiment_core import LABELS,save_json
out=Path('results/model_validation');out.mkdir(exist_ok=True);f=pd.read_csv('data/processed/known_windows.csv');n=len(f);truth=f.type.map(dict(zip(LABELS,range(3)))).to_numpy();groups=f.group.unique();frames=[];sources=[]

def add(path,recipe=None):
    data=pd.read_csv(path);sources.append(path)
    if 'prediction' in data:data=data.rename(columns={'prediction':'predicted_type'})
    if 'recipe' not in data:data['recipe']=recipe
    elif recipe:data['recipe']=recipe+'/'+data.recipe
    for name,g in data.groupby('recipe'):
        assert g.structure_id.nunique()==len(g)==n
        order=g.set_index('structure_id').loc[f.structure_id];assert np.array_equal(order.fold.to_numpy(),pd.read_csv('results/classification_cv/out_of_fold_predictions.csv').set_index('structure_id').loc[f.structure_id,'fold'].to_numpy());frames.append((name,order.predicted_type.map(dict(zip(LABELS,range(3)))).to_numpy()))
add('results/classification_cv/out_of_fold_predictions.csv','original_cnn');add('results/classification_balanced/out_of_fold_predictions.csv','balanced');add('results/classification_shape/out_of_fold_predictions.csv','shape');add('results/classification_kernel/out_of_fold_predictions.csv','image_kernel');add('results/classification_hybrid/out_of_fold_predictions.csv','hybrid_gate');add('results/classification_fine/out_of_fold_predictions.csv','fine')
add('results/classification_span/out_of_fold_predictions.csv','span')
frames.sort(key=lambda p:(p[0]!='original_cnn',p[0]));names=[v[0] for v in frames];confusions=np.zeros((len(frames),len(groups),3,3),int)
for model,(_,pred) in enumerate(frames):
    for group,g in enumerate(groups):
        ids=f.group.eq(g).to_numpy();confusions[model,group]=np.bincount(truth[ids]*3+pred[ids],minlength=9).reshape(3,3)

def metrics(cm):
    tp=np.diagonal(cm,axis1=-2,axis2=-1);true=cm.sum(-1);pred=cm.sum(-2);accuracy=tp.sum(-1)/cm.sum((-2,-1));f1=(2*tp/np.maximum(true+pred,1)).mean(-1);recall=tp[...,0]/np.maximum(true[...,0],1);precision=tp[...,0]/np.maximum(pred[...,0],1)
    return np.stack([accuracy,f1,recall,precision],-1)
observed=metrics(confusions.sum(1));weights=np.random.default_rng(772).multinomial(len(groups),np.full(len(groups),1/len(groups)),size=2000);boot=metrics(np.einsum('bg,mgij->bmij',weights,confusions,optimize=False));difference=boot-boot[:,:1];columns=['accuracy','macro_f1','chid_recall','chid_precision'];rows=[];intervals=[]
for i,name in enumerate(names):
    row=dict(recipe=name,n_loci=n,n_groups=len(groups));row.update({c:float(observed[i,j]) for j,c in enumerate(columns)});rows.append(row);intervals.append(dict(recipe=name,absolute_group95={c:np.quantile(boot[:,i,j],[.025,.975]).tolist() for j,c in enumerate(columns)},delta_vs_original_group95={c:np.quantile(difference[:,i,j],[.025,.975]).tolist() for j,c in enumerate(columns)}))
pd.DataFrame(rows).to_csv(out/'classification_comparison.csv',index=False);save_json(out/'classification_intervals.json',intervals);save_json(out/'sources.json',{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources});save_json(out/'protocol.json',dict(n_group_bootstrap=2000,unit='56maximum-window overlap groups; paired models on identical344loci',primary_new_recipe='fine/fine_consistent_ensemble',model_selection='No overall-test-score selection of primary recipe; ablations retained',limitations=['Multiple exploratory recipes on previously observed test locations; cannot claim independent generalization','Only26CHID labels; macroF1, accuracy and recall have different tradeoffs','Bootstrap resamples spatial groups, not independent biological studies'],report_generation='Paused at user request; no document authoring'))
a=names.index('span/span_auxiliary_ensemble');b=names.index('span/matched_control_ensemble');save_json(out/'span_matched_comparison.json',{c:dict(delta=float(observed[a,j]-observed[b,j]),group95=np.quantile(boot[:,a,j]-boot[:,b,j],[.025,.975]).tolist()) for j,c in enumerate(columns)})
print(pd.DataFrame(rows).to_string(index=False),flush=True)

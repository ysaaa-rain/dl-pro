"""Paired spatial uncertainty; compare frozen recipes without test-based reselection."""
from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
from dlpro.experiment_core import LABELS,save_json
out=Path('results/performance_validation');out.mkdir(exist_ok=True);rng=np.random.default_rng(950);intervals=[]
def difference(frame,group,column,name):
 groups=frame[group].unique();stats=frame.groupby(group)[column].agg(['sum','count']).loc[groups];w=rng.multinomial(len(groups),np.full(len(groups),1/len(groups)),size=2000);boot=np.einsum('bg,g->b',w,stats['sum'].to_numpy(),optimize=False)/np.einsum('bg,g->b',w,stats['count'].to_numpy(),optimize=False)
 return dict(comparison=name,n_cases=len(frame),n_groups=len(groups),mean=float(frame[column].mean()),group95=np.quantile(boot,[.025,.975]).tolist())
native=pd.read_csv('results/sr_data_controls/native_metrics.csv');shape=pd.read_csv('results/sr_data_controls/shape_metrics.csv')
for kind,data,keys,metrics in [('native',native,['sample_index','group','replicate'],['psnr','ssim']),('shape',shape,['structure_id','type','group','replicate'],['oe_ssim_100bp'])]:
 baseline=data[data.method=='projected_bicubic'].set_index(keys)
 for recipe in ['sparse_matched','dense_matched','dense_multiscale']:
  selected=data[data.method.str.startswith(recipe+'_seed')].groupby(keys)[metrics].mean();assert len(selected)==len(baseline)
  delta=(selected-baseline[metrics]).reset_index()
  for metric in metrics:
   intervals.append(difference(delta,'group',metric,f'{recipe}_mean_seed_metric_minus_projected_bicubic_{kind}_{metric}'))
   if kind=='native':
    for repeat,g in delta.groupby('replicate'):intervals.append(difference(g,'group',metric,f'{recipe}_minus_projected_bicubic_rep{repeat}_{metric}'))
 # Training-data hypothesis compared with matched sparse arm.
 sparse=data[data.method.str.startswith('sparse_matched_seed')].groupby(keys)[metrics].mean();dense=data[data.method.str.startswith('dense_matched_seed')].groupby(keys)[metrics].mean();delta=(dense-sparse).reset_index()
 for metric in metrics:intervals.append(difference(delta,'group',metric,f'dense_matched_minus_sparse_matched_{kind}_{metric}'))
save_json(out/'sr_intervals.json',intervals)
if '--sr-only' in sys.argv:
 print(json.dumps(intervals,indent=2),flush=True);raise SystemExit(0)
known=pd.read_csv('data/processed/known_windows.csv');new=pd.read_csv('results/classification_repeat_ssl/out_of_fold_predictions.csv');old=pd.read_csv('results/classification_fine/out_of_fold_predictions.csv');old=old[old.recipe=='fine_consistent_ensemble'].copy();old['recipe']='fixed_fine_baseline';all_rows=pd.concat([new,old]);names=sorted(all_rows.recipe.unique());truth=known.type.map(dict(zip(LABELS,range(3)))).to_numpy();groups=known.group.unique();cm=np.zeros((len(names),len(groups),3,3),int)
for m,name in enumerate(names):
 g=all_rows[all_rows.recipe==name].set_index('structure_id').loc[known.structure_id];assert len(g)==344;pred=g.predicted_type.map(dict(zip(LABELS,range(3)))).to_numpy()
 for j,group in enumerate(groups):
  ids=known.group.eq(group).to_numpy();cm[m,j]=np.bincount(3*truth[ids]+pred[ids],minlength=9).reshape(3,3)
def metrics(c):
 tp=np.diagonal(c,axis1=-2,axis2=-1);true=c.sum(-1);pred=c.sum(-2);f1=2*tp/np.maximum(true+pred,1);return np.stack([tp.sum(-1)/c.sum((-2,-1)),f1.mean(-1),tp[...,0]/np.maximum(pred[...,0],1),tp[...,0]/np.maximum(true[...,0],1),f1[...,0]],-1)
w=rng.multinomial(len(groups),np.full(len(groups),1/len(groups)),size=2000);boot=metrics(np.einsum('bg,mgij->bmij',w,cm,optimize=False));observed=metrics(cm.sum(1));columns=['accuracy','macro_f1','chid_precision','chid_recall','chid_f1'];rows=[]
for name in names:
 i=names.index(name);rows.append(dict(recipe=name,**{c:float(observed[i,j]) for j,c in enumerate(columns)}))
pd.DataFrame(rows).to_csv(out/'classification_comparison.csv',index=False);class_intervals=[]
for a,b in [('repeat_pretrained_raw','matched_control_raw'),('repeat_pretrained_calibrated','matched_control_calibrated'),('validation_selected','fixed_fine_baseline'),('validation_selected','matched_control_calibrated')]:
 ia,ib=names.index(a),names.index(b);delta=boot[:,ia]-boot[:,ib];class_intervals.append(dict(comparison=a+'_minus_'+b,delta={c:float(observed[ia,j]-observed[ib,j]) for j,c in enumerate(columns)},group95={c:np.quantile(delta[:,j],[.025,.975]).tolist() for j,c in enumerate(columns)}))
save_json(out/'classification_intervals.json',class_intervals);save_json(out/'protocol.json',dict(bootstrap_samples=2000,sr_unit='Eight100kbblocks, bothrepeats sharegroup; per-recipe mean ofthree seed metrics, not image ensemble',classification_unit='56maximum-input overlapgroups; original344loci',no_test_based_model_selection=True,limitations=['Retrospective same-study comparisons','Three seeds do not increase biological sample size','Only26CHIDloci','Shape confidence and classification must accompany native gains; no single-score success criterion']))
print(pd.DataFrame(rows).to_string(index=False));print(json.dumps(intervals,indent=2),flush=True)

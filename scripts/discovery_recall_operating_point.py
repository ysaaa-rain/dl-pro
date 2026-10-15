"""Training-only90% recall calibration, with overlap policy and matched random controls."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from dlpro.experiment_core import LENGTH,save_json
out=Path('results/discovery_operating_point');out.mkdir(exist_ok=True);known=pd.read_csv('data/processed/known_windows.csv');folds=pd.read_csv('results/classification_cv/out_of_fold_predictions.csv').set_index('structure_id').loc[known.structure_id,'fold'].to_numpy();scan=pd.read_csv('data/processed/scan_windows.csv');centers=scan.center.to_numpy();distance=abs(centers[:,None]-known.center.to_numpy()[None]);coverage=np.minimum(distance,LENGTH-distance)<=6400;protocol=json.loads(Path('results/discovery_support_cv/protocol.json').read_text());records=[];candidate_records=[];curves=[];random_records=[];rng=np.random.default_rng(645)

def priority(score,separation):
 chosen=[];available=np.ones(len(scan),bool)
 for i in np.argsort(-score,kind='stable'):
  if not available[i]:continue
  chosen.append(i);delta=abs(centers-centers[i]);available&=np.minimum(delta,LENGTH-delta)>=separation
 return np.asarray(chosen)
for fold in range(5):
 z=np.load(f'results/discovery_support_cv/fold{fold}_parameters.npz')['standardized'];choice=protocol['fold_protocols'][fold]['choice'];score=z.max(1) if choice=='maximum' else z.mean(1) if choice=='mean' else np.maximum(z,0).mean(1);test=folds==fold;val=folds==(fold+1)%5;train=~(test|val)
 for separation in [6400,12800]:
  order=priority(score,separation);hits=np.maximum.accumulate(coverage[order],axis=0);train_recalls=hits[:,train].mean(1);eligible=np.flatnonzero(train_recalls>=.9);budget=int(eligible[0]+1) if len(eligible) else len(order);selected=order[:budget];achieved=bool(len(eligible));entry=dict(fold=fold,separation_bp=separation,n_candidates=budget,target_train_recall=.9,target_achieved=achieved,train_recall=float(hits[budget-1,train].mean()),val_recall=float(hits[budget-1,val].mean()),test_recall=float(hits[budget-1,test].mean()));records.append(entry)
  for budget_curve in [32,64,128,192,256,384,len(order)]:
   count=min(budget_curve,len(order));curves.append(dict(fold=fold,separation_bp=separation,n_candidates=count,train_recall=float(hits[count-1,train].mean()),test_recall=float(hits[count-1,test].mean())))
  for rank,i in enumerate(selected):candidate_records.append(dict(fold=fold,separation_bp=separation,rank=rank+1,scan_index=int(i),center=int(centers[i]),score=float(score[i])))
  trials=[]
  for trial in range(500):
   null_order=priority(rng.random(len(scan)),separation)
   if len(null_order)<budget:continue
   null_hit=coverage[null_order[:budget]].any(0);trials.append(null_hit);random_records.append(dict(fold=fold,separation_bp=separation,trial=trial,n_candidates=budget,test_recall=float(null_hit[test].mean())))
  null=np.asarray(trials);assert len(null)>0;entry.update(random_feasible_trials=len(null),random_test_mean=float(null[:,test].mean()))
  for i in np.flatnonzero(test):entry_key=dict(structure_id=known.structure_id.iloc[i],type=known.type.iloc[i],group=int(known.group.iloc[i]),fold=fold,separation_bp=separation,n_candidates=budget,recalled=bool(hits[budget-1,i]),random_hit_probability=float(null[:,i].mean()));candidate_records.append({**entry_key,'record_type':'known_recall'})
  print(entry,flush=True)
rows=pd.DataFrame(candidate_records);recall=rows[rows.record_type.eq('known_recall')].copy();rows[rows.record_type.isna()].drop(columns=['record_type','structure_id','type','group','n_candidates','recalled','random_hit_probability'],errors='ignore').to_csv(out/'candidates.csv',index=False);recall[['structure_id','type','group','fold','separation_bp','n_candidates','recalled','random_hit_probability']].to_csv(out/'out_of_fold_recall.csv',index=False);pd.DataFrame(records).to_csv(out/'fold_metrics.csv',index=False);pd.DataFrame(curves).drop_duplicates(['fold','separation_bp','n_candidates']).to_csv(out/'recall_curve.csv',index=False);pd.DataFrame(random_records).to_csv(out/'random_trials.csv',index=False);summary=[];rng=np.random.default_rng(942)
for separation,g in recall.groupby('separation_bp'):
 assert len(g)==344;groups=g.group.unique();draw=[]
 for _ in range(2000):
  ids=np.concatenate([np.flatnonzero(g.group.to_numpy()==v) for v in rng.choice(groups,len(groups),replace=True)]);draw.append((g.recalled.to_numpy()[ids].astype(float)-g.random_hit_probability.to_numpy()[ids]).mean())
 summary.append(dict(separation_bp=int(separation),center_recall=float(g.recalled.mean()),random_mean=float(g.random_hit_probability.mean()),candidate_counts_by_fold=[r['n_candidates'] for r in records if r['separation_bp']==separation],training_target_reached_folds=sum(r['target_achieved'] for r in records if r['separation_bp']==separation),delta_group95=np.quantile(draw,[.025,.975]).tolist()))
save_json(out/'summary.json',summary);save_json(out/'protocol.json',dict(primary_separation_bp=6400,control_separation_bp=12800,window_width_bp=12800,target_training_recall=.9,score='Frozen support-aware fold-specific morphology scoring; validation choice unchanged',calibration='Minimum retained candidate count reaching90% center recall on training annotations only; report if unreachable',clustering_independence='Overlapping detection windows must not be counted as independent members of a new cluster; require25600bp separation within any proposed cluster',limitations=['Partly supervised operating-point calibration, not fully unsupervised discovery','Retrospective existing test positions; same-study spatial groups','Random orders without sufficient NMS cardinality skipped; feasible trial counts reported','Detection overlap permitted for high recall, separate from independent cluster support']))
print(summary,flush=True)

"""Support-aware sensitivity analysis under identical purged spatial folds."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor
from dlpro.experiment_core import save_json,LENGTH,centered_indices
from dlpro.refinement import morphology_features
from dlpro.coverage_support import coarse_support
out=Path('results/discovery_support_cv');out.mkdir(exist_ok=True);known=pd.read_csv('data/processed/known_windows.csv');folds=pd.read_csv('results/classification_cv/out_of_fold_predictions.csv').set_index('structure_id').loc[known.structure_id,'fold'].to_numpy();scan=pd.read_csv('data/processed/scan_windows.csv');raw=np.load('data/processed/scan_rep1.npz')['raw'];support=coarse_support(np.load('data/processed/coverage_rep1.npz')['coverage'],scan.center.to_numpy());valid_rows=support>=.8;center=scan.center.to_numpy();distance=abs(center[:,None]-known.center.to_numpy()[None]);coverage=np.minimum(distance,LENGTH-distance)<=6400;d=abs(np.arange(128)[:,None]-np.arange(128)[None,:]);starts=np.asarray([centered_indices(int(c))[1]*200 for c in center]);ends=starts+25600

def select(score,budget):
 chosen=[];available=np.ones(len(scan),bool)
 for i in np.argsort(-score,kind='stable'):
  if not available[i]:continue
  chosen.append(i);delta=abs(center-center[i]);available&=np.minimum(delta,LENGTH-delta)>=12800
  if len(chosen)==budget:return np.asarray(chosen)
 raise ValueError('Insufficient windows')

records=[];candidate_records=[];protocol=[]
for fold in range(5):
 test=folds==fold;val=folds==(fold+1)%5;held=known[test|val];overlap=(starts[:,None]<held.window_end.to_numpy()[None])&(ends[:,None]>held.window_start.to_numpy()[None]);eligible=scan.training_background.to_numpy()&~overlap.any(1)&(starts>=0)&(ends<=LENGTH);blocks=scan.loc[eligible,'background_block'].unique();order=np.random.default_rng(42+fold).permutation(blocks);train=eligible&scan.background_block.isin(order[:max(1,int(.7*len(order)))]).to_numpy();assert not overlap[train].any()
 mask=valid_rows[:,:,:,None]&valid_rows[:,:,None,:];expected=np.zeros((2,128))
 for channel in range(2):
  for delta in range(128):
   a=raw[train,channel][:,d==delta];m=mask[train,channel][:,d==delta];expected[channel,delta]=max(float(a[m].mean()) if m.any() else 0,1e-8)
 features=morphology_features(raw,expected,valid_rows);density=np.log(np.maximum((raw[:,0]*mask[:,0]).sum((1,2))/np.maximum(mask[:,0].sum((1,2)),1),1e-12));z=np.empty_like(features);calibration=[]
 for column in range(features.shape[1]):
  response=np.log(np.maximum(features[:,column],1e-6));fit=HuberRegressor().fit(density[train,None],response[train]);residual=response[train]-fit.predict(density[train,None]);mu=float(np.median(residual));spread=max(float(1.4826*np.median(abs(residual-mu))),.05);z[:,column]=(response-fit.predict(density[:,None])-mu)/spread;calibration.append(dict(feature=column,slope=float(fit.coef_[0]),intercept=float(fit.intercept_),location=mu,spread=spread))
 alternatives={'maximum':z.max(1),'mean':z.mean(1),'positive_mean':np.maximum(z,0).mean(1)};validation={m:float(coverage[select(score,64)].any(0)[val].mean()) for m,score in alternatives.items()};choice=max(alternatives,key=lambda m:validation[m]);score=alternatives[choice]
 for budget in [64,128]:
  ids=select(score,budget);hit=coverage[ids].any(0)
  for i in np.flatnonzero(test):records.append(dict(structure_id=known.structure_id.iloc[i],type=known.type.iloc[i],group=int(known.group.iloc[i]),fold=fold,budget=budget,recalled=bool(hit[i])))
  for rank,i in enumerate(ids):candidate_records.append(dict(fold=fold,budget=budget,rank=rank+1,scan_index=int(i),center=int(center[i]),score=float(score[i])))
 np.savez_compressed(out/f'fold{fold}_parameters.npz',expected=expected,features=features,standardized=z,background_train=train,valid_rows=valid_rows);protocol.append(dict(fold=fold,choice=choice,validation_choices=validation,calibration=calibration,n_background_train=int(train.sum())));print('support fold',fold,'choice',choice,'test64',float(coverage[select(score,64)].any(0)[test].mean()),flush=True)
frame=pd.DataFrame(records);frame.to_csv(out/'out_of_fold_recall.csv',index=False);pd.DataFrame(candidate_records).to_csv(out/'candidates.csv',index=False);null=pd.read_csv('results/discovery_spatial_cv/random_locus_probabilities.csv');baseline=pd.read_csv('results/discovery_spatial_cv/out_of_fold_recall.csv');baseline=baseline[baseline.method=='morphology'];summary=[];rng=np.random.default_rng(547)
for budget,g in frame.groupby('budget'):
 g=g.merge(null,on=['structure_id','budget']).merge(baseline[['structure_id','budget','recalled']].rename(columns={'recalled':'unmasked_recalled'}),on=['structure_id','budget']);assert len(g)==344;groups=g.group.unique();draw=[]
 for _ in range(2000):
  ids=np.concatenate([np.flatnonzero(g.group.to_numpy()==v) for v in rng.choice(groups,len(groups),replace=True)]);draw.append([(g.recalled.to_numpy()[ids].astype(float)-g.random_hit_probability.to_numpy()[ids]).mean(),(g.recalled.to_numpy()[ids].astype(float)-g.unmasked_recalled.to_numpy()[ids].astype(float)).mean()])
 ci=np.quantile(draw,[.025,.975],axis=0);summary.append(dict(budget=int(budget),center_recall=float(g.recalled.mean()),random_mean=float(g.random_hit_probability.mean()),unmasked_recall=float(g.unmasked_recalled.mean()),delta_vs_random_group95=ci[:,0].tolist(),delta_vs_unmasked_group95=ci[:,1].tolist()))
save_json(out/'summary.json',summary);save_json(out/'protocol.json',dict(minimum_observed_native_bp_fraction=.8,scoring_source='WTrep1 coverage only; rep2 not used for scoring or selection',expectation='Distance expectation from eligible training-background pixels whose two bins satisfy support threshold',selection='Same three morphology aggregations selected by cyclic validation64candidate recall',fold_protocols=protocol,limitations=['Operational sensitivity analysis; zero-contact bins are not assigned mapping or biological causes','Previously observed test positions; retrospective spatial evaluation','Coverage is measured chromosome-wide, analogous to library-size preprocessing','Same-study, no independent source validation']))
print(summary,flush=True)

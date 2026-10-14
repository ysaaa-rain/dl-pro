"""Fixed-budget morphology scan; choose score on validation labels, retain negative controls."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import HuberRegressor
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dlpro.experiment_core import ShapeTransform,LENGTH,hits,save_json,configure
from dlpro.refinement import morphology_features
configure(42);out=Path('results/discovery_refined');out.mkdir(exist_ok=True);(out/'heatmaps').mkdir(exist_ok=True)
known=pd.read_csv('data/processed/known_windows.csv');scan=pd.read_csv('data/processed/scan_windows.csv');raw=[np.load(f'data/processed/scan_rep{i}.npz')['raw'] for i in [1,2]]
tf=ShapeTransform.load('results/discovery/transform.npz');features=[morphology_features(r,tf.expected) for r in raw]
blocks=scan.loc[scan.training_background,'background_block'].unique();order=np.random.default_rng(42).permutation(blocks);training=scan.training_background&scan.background_block.isin(order[:int(.7*len(order))])
logdensity=[np.log(np.maximum(r[:,0].mean((1,2)),1e-12)) for r in raw];standard=[];fit_records=[]
for col in range(features[0].shape[1]):
 y=np.log(np.maximum(features[0][:,col],1e-6));fit=HuberRegressor().fit(logdensity[0][training,None],y[training]);resid=y[training]-fit.predict(logdensity[0][training,None]);median=np.median(resid);scale=max(1.4826*np.median(abs(resid-median)),.05)
 fit_records.append(dict(feature=col,slope=float(fit.coef_[0]),intercept=float(fit.intercept_),location=float(median),scale=float(scale)))
 standard.append([(np.log(np.maximum(f[:,col],1e-6))-fit.predict(d[:,None])-median)/scale for f,d in zip(features,logdensity)])
z=[np.stack([r[i] for r in standard],1) for i in [0,1]]
old=pd.read_csv('results/discovery/all_windows.csv');print('old columns',old.columns.tolist(),flush=True)
scores={'coherent_max':z[0].max(1),'coherent_mean':z[0].mean(1),'coherent_positive':np.maximum(z[0],0).mean(1),'density':logdensity[0]}
# Original saved order can be reconstructed from its score.
score_column=next(c for c in ['shape_score_rep1','shape_score','score'] if c in old.columns);scores['first_round_AE']=old[score_column].to_numpy()
centers=scan.center.to_numpy();dist=abs(centers[:,None]-known.center.to_numpy()[None]);coverage=np.minimum(dist,LENGTH-dist)<=6400

def select(score,budget):
 chosen=[];available=np.ones(len(scan),bool)
 for idx in np.argsort(-score,kind='stable'):
  if not available[idx]:continue
  chosen.append(idx);d=abs(centers-centers[idx]);available&=np.minimum(d,LENGTH-d)>=12800
  if len(chosen)==budget:break
 assert len(chosen)==budget
 return np.asarray(chosen)

def recall(ids,split):return float(coverage[ids].any(0)[known.split.eq(split)].mean())
# Primary budget fixed before evaluation; only validation labels choose the morphology combination.
primary=64;choices=['coherent_max','coherent_mean','coherent_positive'];val_metrics={m:recall(select(scores[m],primary),'val') for m in choices};chosen=max(choices,key=lambda m:(val_metrics[m],-choices.index(m)))
rows=[];rng=np.random.default_rng(314);random_records=[]
for budget in [64,128]:
 for method,score in scores.items():
  ids=select(score,budget)
  for split in ['train','val','test']:rows.append(dict(budget=budget,method=method,split=split,center_recall=recall(ids,split)))
 for trial in range(500):
  ids=select(rng.random(len(scan)),budget)
  for split in ['val','test']:random_records.append(dict(budget=budget,trial=trial,split=split,center_recall=recall(ids,split)))
pd.DataFrame(rows).to_csv(out/'budget_metrics.csv',index=False);pd.DataFrame(random_records).to_csv(out/'random_trials.csv',index=False)
selected=select(scores[chosen],primary);pca=PCA(n_components=6,svd_solver='full').fit(z[0][training]);embedding=pca.transform(z[0]);kmeans=KMeans(n_clusters=6,n_init=20,random_state=42).fit(embedding[selected]);triangle=np.triu_indices(128,2);distance=abs(np.arange(128)[:,None]-np.arange(128)[None,:]);expect=tf.expected[0,distance];records=[]
for j,idx in enumerate(selected):
 a=raw[0][idx,0]/expect;b=raw[1][idx,0]/expect;r=float(np.corrcoef(a[triangle],b[triangle])[0,1]);matched=known.loc[hits(int(centers[idx]),12800,known,buffer=200),'structure_id'].tolist()
 records.append(dict(candidate_id=f'R{j+1:04d}',scan_index=int(idx),center=int(centers[idx]),score=float(scores[chosen][idx]),rep2_score=float((z[1].max(1) if chosen=='coherent_max' else z[1].mean(1) if chosen=='coherent_mean' else np.maximum(z[1],0).mean(1))[idx]),cluster=int(kmeans.labels_[j]),repeat_oe_correlation=r,reproducible=np.isfinite(r) and r>=.5,known_overlap=bool(matched),known_ids=';'.join(matched)))
 fig,axes=plt.subplots(1,2,figsize=(7,3),constrained_layout=True);lo=0;hi=np.quantile(np.log1p(np.stack([a,b])),.995)
 for ax,v,label in zip(axes,[a,b],['WT rep1','WT rep2']):ax.imshow(np.log1p(v),vmin=lo,vmax=hi,cmap='magma',origin='lower');ax.set_title(label)
 fig.suptitle(f'R{j+1:04d}; center {centers[idx]}; r={r:.3f}');fig.savefig(out/'heatmaps'/f'R{j+1:04d}.png',dpi=100);plt.close(fig)
candidates=pd.DataFrame(records);candidates.to_csv(out/'candidates.csv',index=False);unannotated=candidates[~candidates.known_overlap&candidates.reproducible];unannotated.to_csv(out/'unannotated_reproducible.csv',index=False)
novel=[]
for label,g in candidates.groupby('cluster'):
 if not g.known_overlap.any() and g.reproducible.sum()>=5:novel.append(int(label))
np.savez_compressed(out/'features.npz',rep1=features[0],rep2=features[1],z1=z[0],z2=z[1],pca_components=pca.components_,pca_mean=pca.mean_,cluster_centers=kmeans.cluster_centers_)
save_json(out/'protocol.json',dict(primary_budget=primary,secondary_budget=128,nms_bp=12800,score_choice=chosen,validation_choices=val_metrics,fit_background_windows=int(training.sum()),feature_recipe='Positive multiscale Gaussian local O/E contrasts in window central half; zero rows and |distance|<2 masked; two scales, sigma1/2/4/8 against sigma12',density_calibration=fit_records,score_density_spearman=float(spearmanr(scores[chosen],logdensity[0]).statistic),n_candidates=len(candidates),unannotated_reproducible=len(unannotated),novel_clusters=novel,replicate_threshold=.5,random_trials=500,limitations=['Known validation labels choose morphology aggregation; partly supervised model selection, not fully unsupervised discovery','Test labels previously examined in first round; retrospective comparison, no fresh independent test','200bp exclusion and >=5 reproducible members required for new cluster','High 200bp repeat correlation does not establish new biology']))
print(pd.DataFrame(rows).to_string(index=False));print('chosen',chosen,'unannotated',len(unannotated),'new clusters',novel,flush=True)

"""Exploratory shape clustering with repeat support and matched-background controls."""
import sys,json,itertools
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist,pdist
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dlpro.experiment_core import ShapeTransform,LENGTH,hits,save_json
from dlpro.coverage_support import coarse_support
from dlpro.novelty import shape_vectors

OUT=Path('results/unannotated_shape_clusters');OUT.mkdir(exist_ok=True)
(OUT/'heatmaps').mkdir(exist_ok=True)
seed=42;rng=np.random.default_rng(seed)
scan=pd.read_csv('data/processed/scan_windows.csv');known=pd.read_csv('data/processed/known_windows.csv')
candidates=pd.read_csv('results/discovery/unannotated_reproducible.csv')
raw={r:np.load(f'data/processed/scan_rep{r}.npz')['raw'] for r in (1,2)}
kr={r:np.load(f'data/processed/known_rep{r}.npz')['raw'] for r in (1,2)}
expected=ShapeTransform.load('results/discovery/transform.npz').expected
support={r:coarse_support(np.load(f'data/processed/coverage_rep{r}.npz')['coverage'],scan.center.to_numpy())>=.8 for r in (1,2)}
ks={r:coarse_support(np.load(f'data/processed/coverage_rep{r}.npz')['coverage'],known.center.to_numpy())>=.8 for r in (1,2)}
vec={r:shape_vectors(raw[r],support[r],expected,16)[0] for r in (1,2)}
kv={r:shape_vectors(kr[r],ks[r],expected,16)[0] for r in (1,2)}
train=scan.training_background.to_numpy(bool)
pca=PCA(n_components=12,svd_solver='full').fit(vec[1][train])
def project(x):
 return np.einsum('nd,kd->nk',x.astype(float)-pca.mean_,pca.components_,optimize=False)
z={r:project(vec[r]) for r in (1,2)};kz={r:project(kv[r]) for r in (1,2)}
if not all(np.isfinite(x).all() for x in [*z.values(),*kz.values()]):raise ValueError('Nonfinite shape representation')
scale=np.maximum(z[1][train].std(0),1e-6)
z={r:z[r]/scale for r in (1,2)};kz={r:kz[r]/scale for r in (1,2)}
triangle=np.triu(np.ones((128,128),bool),2);delta=abs(np.arange(128)[:,None]-np.arange(128)[None,:])
rows=[]
for row in candidates.itertuples():
 i=row.scan_index;valid=triangle&support[1][i,0,:,None]&support[1][i,0,None,:]&support[2][i,0,:,None]&support[2][i,0,None,:]
 a=(raw[1][i,0]/expected[0,delta])[valid];b=(raw[2][i,0]/expected[0,delta])[valid]
 corr=float(np.corrcoef(a,b)[0,1]) if len(a)>2 and a.std()>0 and b.std()>0 else np.nan
 full=bool(hits(int(row.center),25600,known,buffer=200).any())
 nearest=int(np.argmin(np.minimum(abs(known.center.to_numpy()-row.center),LENGTH-abs(known.center.to_numpy()-row.center))))
 rows.append(dict(candidate_id=row.candidate_id,scan_index=i,chrom=row.chrom,start=row.start,end=row.end,center=row.center,length_bp=row.length_bp,shape_score=row.shape_score,repeat_oe_correlation=corr,supported_triangle_fraction=float(valid.sum()/triangle.sum()),known_context_overlap=full,nearest_known_id=known.iloc[nearest].structure_id,nearest_known_center_distance_bp=int(min(abs(known.iloc[nearest].center-row.center),LENGTH-abs(known.iloc[nearest].center-row.center))),heatmap_file=f'heatmaps/{row.candidate_id}.png'))
f=pd.DataFrame(rows)
f['quality_pass']=(f.supported_triangle_fraction>=.5)&(f.repeat_oe_correlation>=.5)
f['eligible']=f.quality_pass&~f.known_context_overlap
accepted=[]
for row in f[f.eligible].sort_values('center').itertuples():
 d=abs(np.asarray(accepted)-row.center)
 if np.all(np.minimum(d,LENGTH-d)>=25600):accepted.append(row.center)
f['independent_member']=f.center.isin(accepted)
ids=f.loc[f.independent_member,'scan_index'].to_numpy(int)
# Distance scales fixed from background geometry, independent of candidate outcomes.
bg=np.flatnonzero(train)
bgd=pdist(z[1][bg]);eps_values=np.quantile(bgd,[.05,.10,.20])
settings=list(itertools.product(range(3),(2,3)))
labels={};clusters=[];cluster_centers={}
# Known-reference novelty uses training known references; held-out distances set a descriptive boundary.
k_train=known.split.eq('train').to_numpy();k_val=known.split.ne('train').to_numpy()
novelty_threshold=float(np.quantile(cdist(kz[1][k_val],kz[1][k_train]).min(1),.95))
known_dist={r:cdist(z[r],kz[1][k_train]).min(1) for r in (1,2)}
def cluster_stat(points,eps,min_samples):
 lab=DBSCAN(eps=float(eps),min_samples=min_samples).fit_predict(points)
 groups=[np.flatnonzero(lab==g) for g in set(lab) if g>=0]
 return lab,groups
# Control the largest apparent cluster over the complete six-setting search.
null_max=[]
if len(ids)>=2:
 density=np.log(np.maximum(raw[1][:,0].mean((1,2)),1e-12))
 for draw in range(200):
  pool=[]
  for i in ids:
   eligible=bg[abs(density[bg]-density[i])<=.5]
   if len(eligible)<5:eligible=bg
   options=rng.permutation(eligible)
   chosen=None
   for j in options:
    d=abs(scan.center.to_numpy()[pool]-scan.center.iloc[j])
    if j not in pool and np.all(np.minimum(d,LENGTH-d)>=25600):chosen=int(j);break
   if chosen is None:break
   pool.append(chosen)
  if len(pool)!=len(ids):continue
  sizes=[]
  for ei,minimum in settings:
   _,groups=cluster_stat(z[1][pool],eps_values[ei],minimum)
   sizes.extend([len(g) for g in groups])
  null_max.append(max(sizes,default=0))
 for ei,minimum in settings:
  name=f'q{[5,10,20][ei]}_m{minimum}'
  lab,groups=cluster_stat(z[1][ids],eps_values[ei],minimum);labels[name]=lab
  lab2,_=cluster_stat(z[2][ids],eps_values[ei],minimum)
  for ci,members in enumerate(groups):
   ix=ids[members];cluster_centers[f'{name}_cluster{int(lab[members[0]])}']=z[1][ix].mean(0);repeat_stable=bool(np.all(lab2[members]>=0) and len(set(lab2[members]))==1)
   novel=bool(np.all(known_dist[1][ix]>novelty_threshold)&np.all(known_dist[2][ix]>novelty_threshold))
   tail=(1+sum(n>=len(ix) for n in null_max))/(1+len(null_max))
   qualifies=repeat_stable and novel and tail<=.05
   clusters.append(dict(setting=name,cluster=int(lab[members[0]]),members=len(ix),member_ids=';'.join(f.set_index('scan_index').loc[ix,'candidate_id']),repeat_cluster_consistent=repeat_stable,all_members_beyond_known_reference=novel,search_adjusted_background_tail=tail,tentative_new_shape_cluster=qualifies,representative_id=f.set_index('scan_index').loc[ix[0],'candidate_id']))
# Primary output uses the middle background-distance scale and two independent members.
primary='q10_m2';f['cluster']=-1
if primary in labels:f.loc[f.independent_member,'cluster']=labels[primary]
f['known_structure_distinction']=np.where(f.known_context_overlap,'Full 25.6kb context intersects known annotation; excluded from new-cluster inference','Full 25.6kb context plus 200bp buffer has no known annotation; morphological novelty separately tested')
f['evidence_status']=np.select([~f.quality_pass,f.known_context_overlap,~f.independent_member],['repeat/support insufficient','known-context proximity','spatially redundant'],default='independent unannotated reproducible candidate; novelty unconfirmed')
f.to_csv(OUT/'candidate_audit.csv',index=False)
f[f.independent_member].to_csv(OUT/'candidate_new_structures.csv',index=False)
pd.DataFrame(clusters,columns=['setting','cluster','members','member_ids','repeat_cluster_consistent','all_members_beyond_known_reference','search_adjusted_background_tail','tentative_new_shape_cluster','representative_id']).to_csv(OUT/'cluster_evidence.csv',index=False)
np.savez_compressed(OUT/'shape_representation.npz',rep1=z[1],rep2=z[2],pca_components=pca.components_,pca_mean=pca.mean_,latent_scale=scale,known_rep1=kz[1])
np.savez_compressed(OUT/'cluster_centers.npz',**cluster_centers)
np.savez_compressed(OUT/'matched_background_control.npz',maximum_cluster_members=null_max,eps_values=eps_values)
# Fixed rendering shared by every candidate and both replicates.
for row in f.itertuples():
 fig,axes=plt.subplots(1,2,figsize=(8,3.7),constrained_layout=True)
 for ax,r in zip(axes,(1,2)):
  a=raw[r][row.scan_index,0]/expected[0,delta];mask=delta>=2
  field=np.log1p(a/max(float(a[mask].mean()),1e-12));field=np.ma.masked_where(~mask,field)
  im=ax.imshow(field,origin='lower',cmap='magma',vmin=0,vmax=2,extent=[row.start/1000,row.end/1000]*2)
  ax.set(title=f'WT rep{r}',xlabel='Genome coordinate (kb)')
 fig.colorbar(im,ax=axes,label='Amplitude-normalized log(1+O/E)')
 fig.suptitle(f'{row.candidate_id}; supported O/E r={row.repeat_oe_correlation:.3f}')
 fig.savefig(OUT/row.heatmap_file,dpi=130);plt.close(fig)
fig,ax=plt.subplots(figsize=(7,5));ax.scatter(z[1][bg,0],z[1][bg,1],s=10,c='gray',alpha=.3,label='Unannotated background')
ax.scatter(kz[1][:,0],kz[1][:,1],s=12,c='steelblue',label='Known annotations')
if len(ids):ax.scatter(z[1][ids,0],z[1][ids,1],s=35,c='darkorange',label='Independent candidates')
ax.set(xlabel='Standardized PC1',ylabel='Standardized PC2');ax.legend();fig.tight_layout();fig.savefig(OUT/'shape_embedding.png',dpi=140);plt.close(fig)
summary=dict(input_candidates=len(f),quality_pass=int(f.quality_pass.sum()),unannotated_full_context=int(f.eligible.sum()),independent_candidates=len(ids),cluster_tests=len(clusters),qualified_cluster_tests=sum(c['tentative_new_shape_cluster'] for c in clusters),background_draws=len(null_max),known_reference_distance_threshold=novelty_threshold)
save_json(OUT/'summary.json',summary)
save_json(OUT/'protocol.json',dict(seed=seed,source_candidates='results/discovery/unannotated_reproducible.csv',input_scales_bp=[100,200],pooled_bins=16,pca_components=12,representation_fit='WT rep1 unannotated training backgrounds only',repeat_threshold=.5,common_supported_triangle_min=.5,native_support_fraction=.8,known_exclusion_context_bp=25600,known_buffer_bp=200,independent_spacing_bp=25600,cluster_method='DBSCAN',eps_quantiles=[.05,.10,.20],minimum_members=[2,3],primary_setting=primary,novelty_rule='Every member in both repeats exceeds 95th percentile held-out known-to-training-known distance',control='200 density-matched spatially separated background sets; maximum cluster size across all six settings',heatmap_vmin=0,heatmap_vmax=2,limitations=['Retrospective exploratory comparison, not independent validation','Background may contain unannotated structures','Cluster-size background tail is a diagnostic for this search, not a biological discovery p-value','Window coordinates are scan extents, not inferred structural boundaries']))
print(json.dumps(summary,indent=2));print(pd.DataFrame(clusters).to_string(index=False))

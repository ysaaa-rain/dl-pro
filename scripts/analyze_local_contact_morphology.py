"""Fixed local morphology evaluation on known structures, candidates and background."""
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist,pdist,squareform
from scipy.stats import spearmanr
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import DBSCAN
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dlpro.local_morphology import analyze,normalized_field,interval_iou,FEATURE_NAMES
from dlpro.experiment_core import LENGTH,save_json
from dlpro.coverage_support import coarse_support

OUT=Path('results/local_contact_morphology');OUT.mkdir(exist_ok=True)
(OUT/'heatmaps').mkdir(exist_ok=True)
SCALES=[50,100,200];SEED=42

def support_at(coverage,centers,bp):
 ids=np.arange(len(coverage));width=np.minimum(10,LENGTH-10*ids);b=ids//(bp//10);total=(LENGTH+bp-1)//bp
 frac=np.bincount(b,weights=width*(coverage>0),minlength=total)/np.bincount(b,weights=width,minlength=total)
 starts=np.floor((centers+bp/2)/bp).astype(int)-64
 return frac[(starts[:,None]+np.arange(128))%total]>=.8

def project(pca,x):
 return np.einsum('nd,kd->nk',x-pca.mean_,pca.components_,optimize=False)

def feature_routes(arrays):
 return {'fixed_geometry':arrays['fixed_geometry'].reshape(len(arrays['fixed_geometry']),-1),
         'localized_geometry':arrays['geometry'].reshape(len(arrays['geometry']),-1),
         'aligned_image':arrays['images'][:,1:].reshape(len(arrays['images']),-1)}

def fit_route(method,one,two,fit_ids):
 if method=='aligned_image':
  pca=PCA(n_components=16,svd_solver='full',random_state=SEED).fit(one[fit_ids])
  one=project(pca,one);two=project(pca,two)
 scaler=RobustScaler(quantile_range=(10,90)).fit(one[fit_ids])
 a=scaler.transform(one);b=scaler.transform(two)
 if not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Nonfinite standardized morphology')
 return a,b

def main():
 known=pd.read_csv('data/processed/known_windows.csv');scan=pd.read_csv('data/processed/scan_windows.csv')
 candidate=pd.read_csv('results/unannotated_shape_clusters/candidate_new_structures.csv')
 nknown=len(known);nscan=len(scan);centers=np.r_[known.center,scan.center]
 ids=list(known.structure_id)+[f'scan_{i}' for i in range(nscan)]
 folds=pd.read_csv('results/classification_cv/out_of_fold_predictions.csv').set_index('structure_id').loc[known.structure_id,'fold'].to_numpy()
 trainbg=scan.training_background.to_numpy(bool)
 cache=OUT/'measurements.npz';measurement=OUT/'per_window_geometry.csv'
 signature=hashlib.sha256(Path('src/dlpro/local_morphology.py').read_bytes()).hexdigest()
 signature_path=OUT/'measurement_signature.json'
 save_json(OUT/'protocol.json',dict(seed=SEED,scales_bp=SCALES,localization_width_bins=[8,16,24,32,48,64,96],localization_stride_bins=4,
  localization='Support-aware positive Gaussian-bandpass energy / valid-pair-count^0.75; independent localization in each repeat; exhaustive same search for known/candidate/background',
  normalization='Per-scale WT rep1 distance expectation from unannotated background; window amplitude normalized; no class labels used for localization',
  representation={'geometric_features':list(FEATURE_NAMES),'aligned_pixels':[32,32],'fixed_control':'same geometry on entire input'},
  evaluation='Existing spatial five folds; train WT rep1, evaluate both repeats; fold-specific scaler and image PCA; whole normalization fit globally on old background, so evaluation is retrospective and not fully purged preprocessing',
  novelty='Leave one known type out of reference set; nearest allowed-type distance; assess held spatial fold against allowed types and eligible background separately',
  candidate_selection='Existing 16 independent candidates frozen before local analysis',
  interpretation='Window footprint bounds are estimated not validated true biological boundaries; density/support/distance matched controls run same exhaustive localization',
  limitations=['No independent new data','Known annotations can be nested','Same repeats observed previously','Matched background reference is descriptive, not a significance claim']))
 if not cache.exists() or not signature_path.exists() or json.loads(signature_path.read_text()).get('module_sha256')!=signature:
  arrays={r:{'geometry':np.zeros((len(ids),3,len(FEATURE_NAMES))),'fixed_geometry':np.zeros((len(ids),3,len(FEATURE_NAMES))),
   'images':np.zeros((len(ids),3,32,32),dtype='float32'),'masks':np.zeros((len(ids),3,32,32),dtype=bool),'bounds':np.zeros((len(ids),3,2),int)} for r in (1,2)}
  records=[];expected_scales={}
  for channel,bp in enumerate(SCALES):
   raw={};supp={}
   for r in (1,2):
    if bp==50:
     a=np.load(f'data/processed/fine_rep{r}.npz')['raw'][:,0];b=np.load(f'data/processed/fine_scan_rep{r}.npz')['raw'][:,0]
    else:
     ch=0 if bp==100 else 1
     a=np.load(f'data/processed/known_rep{r}.npz')['raw'][:,ch];b=np.load(f'data/processed/scan_rep{r}.npz')['raw'][:,ch]
    raw[r]=np.concatenate([a,b]);coverage=np.load(f'data/processed/coverage_rep{r}.npz')['coverage'];supp[r]=support_at(coverage,centers,bp)
   d=abs(np.arange(128)[:,None]-np.arange(128)[None,:]);expected=np.ones(128)
   fit=raw[1][nknown:][trainbg];sr=supp[1][nknown:][trainbg]
   for distance in range(128):
    y,x=np.where(d==distance);valid=sr[:,y]&sr[:,x];vals=fit[:,y,x]
    expected[distance]=max(float(vals[valid].mean()) if valid.any() else 0,1e-12)
   expected_scales[str(bp)]=expected.tolist()
   for r in (1,2):
    for i,matrix in enumerate(raw[r]):
     result=analyze(matrix,expected,supp[r][i],bp);region=result['region']
     arrays[r]['geometry'][i,channel]=result['geometry'];arrays[r]['fixed_geometry'][i,channel]=result['fixed_geometry'];arrays[r]['images'][i,channel]=result['image'];arrays[r]['masks'][i,channel]=result['image_support']
     window_start=int(np.floor((centers[i]+bp/2)/bp)-64)*bp
     bounds=[window_start+region['lo']*bp,window_start+region['hi']*bp];arrays[r]['bounds'][i,channel]=bounds
     mask=result['valid']&np.triu(np.ones((128,128),bool),1)
     row=dict(item_id=ids[i],item_index=i,actual_type=known.type.iloc[i] if i<nknown else 'unannotated',repeat=r,bin_bp=bp,
       estimated_start=bounds[0],estimated_end=bounds[1],estimated_length_bp=bounds[1]-bounds[0],localization_score=region['score'],
       local_support_fraction=region['support_fraction'],valid_localization=region['valid'],
       log_density=float(np.log(max(float(matrix[mask].mean()) if mask.any() else 0,1e-12))),**dict(zip(FEATURE_NAMES,result['geometry'])))
     records.append(row)
    print('geometry scale',bp,'repeat',r,'windows',len(ids),flush=True)
  pd.DataFrame(records).to_csv(measurement,index=False)
  np.savez_compressed(cache,**{f'{key}_rep{r}':value for r,values in arrays.items() for key,value in values.items()})
  save_json(OUT/'distance_expectation.json',expected_scales)
  save_json(signature_path,dict(module_sha256=signature,raw_inputs='Existing WT scan/known/fine NPZs; metadata in data/processed/*sources.json'))
 else:
  stored=np.load(cache);arrays={r:{key:stored[f'{key}_rep{r}'] for key in ['geometry','fixed_geometry','images','masks','bounds']} for r in (1,2)}
 table=pd.read_csv(measurement)
 # Stability includes separate estimated positions, never selected jointly from repeats.
 stability=[]
 for i,item in enumerate(ids):
  for ch,bp in enumerate(SCALES):
   b1=arrays[1]['bounds'][i,ch];b2=arrays[2]['bounds'][i,ch]
   common=arrays[1]['masks'][i,ch]&arrays[2]['masks'][i,ch]&np.triu(np.ones((32,32),bool),1)
   v1=arrays[1]['images'][i,ch][common];v2=arrays[2]['images'][i,ch][common]
   local_corr=float(np.corrcoef(v1,v2)[0,1]) if len(v1)>3 and v1.std()>0 and v2.std()>0 else 0.
   row=dict(item_id=item,actual_type=known.type.iloc[i] if i<nknown else 'unannotated',bin_bp=bp,repeat_boundary_iou=interval_iou(b1,b2),repeat_center_shift_bp=float(abs(b1.mean()-b2.mean())),
      aligned_image_repeat_correlation=local_corr)
   if i<nknown:
    row['rep1_annotated_interval_iou']=interval_iou(b1,(known.start.iloc[i],known.end.iloc[i]))
    row['rep2_annotated_interval_iou']=interval_iou(b2,(known.start.iloc[i],known.end.iloc[i]))
   stability.append(row)
 stability=pd.DataFrame(stability);stability.to_csv(OUT/'localization_stability.csv',index=False)
 stability[stability.actual_type!='unannotated'].groupby(['actual_type','bin_bp']).median(numeric_only=True).to_csv(OUT/'known_localization_summary.csv')
 routes={r:feature_routes(arrays[r]) for r in (1,2)}
 classification=[];novelty=[]
 starts=np.floor((scan.center.to_numpy()+100)/200).astype(int)*200-12800;ends=starts+25600
 for fold in range(5):
  test=folds==fold;train=~test
  held=known[test]
  overlap=((starts[:,None]<held.window_end.to_numpy()[None])&(ends[:,None]>held.window_start.to_numpy()[None])).any(1)
  bgfit=np.flatnonzero(trainbg&~overlap)+nknown;bgtest=np.flatnonzero(trainbg&overlap)+nknown
  training=np.r_[np.flatnonzero(train),bgfit]
  for method in routes[1]:
   a,b=fit_route(method,routes[1][method],routes[2][method],training)
   forest=RandomForestClassifier(n_estimators=120,max_depth=5,min_samples_leaf=3,class_weight='balanced_subsample',random_state=SEED,n_jobs=1).fit(a[:nknown][train],known.type[train])
   for r,vectors in [(1,a),(2,b)]:
    pred=forest.predict(vectors[:nknown][test])
    for i,prediction in zip(np.flatnonzero(test),pred):classification.append(dict(method=method,fold=fold,repeat=r,structure_id=known.structure_id.iloc[i],actual=known.type.iloc[i],predicted=prediction,group=int(known.group.iloc[i])))
   for hidden in ['CHIN','OPCID','CHID']:
    allowed=train&known.type.ne(hidden).to_numpy()
    novel_fit=np.r_[np.flatnonzero(allowed),bgfit]
    na,nb=fit_route(method,routes[1][method],routes[2][method],novel_fit)
    reference=na[:nknown][allowed]
    for r,vectors in [(1,na),(2,nb)]:
     score=cdist(vectors,reference).min(1)/np.sqrt(vectors.shape[1])
     for i in np.flatnonzero(test):novelty.append(dict(method=method,fold=fold,repeat=r,hidden_type=hidden,item_id=ids[i],actual_type=known.type.iloc[i],score=score[i],is_hidden=known.type.iloc[i]==hidden))
     for i in bgtest:novelty.append(dict(method=method,fold=fold,repeat=r,hidden_type=hidden,item_id=ids[i],actual_type='background',score=score[i],is_hidden=False))
  print('validation fold',fold,flush=True)
 predictions=pd.DataFrame(classification);predictions.to_csv(OUT/'known_classification_predictions.csv',index=False)
 metrics=[]
 for (method,repeat),part in predictions.groupby(['method','repeat']):
  metrics.append(dict(method=method,repeat=int(repeat),accuracy=accuracy_score(part.actual,part.predicted),macro_f1=f1_score(part.actual,part.predicted,average='macro')))
 pd.DataFrame(metrics).to_csv(OUT/'known_classification_summary.csv',index=False)
 ns=pd.DataFrame(novelty);ns.to_csv(OUT/'held_type_novelty_scores.csv',index=False)
 auc=[]
 for (method,repeat,hidden),part in ns.groupby(['method','repeat','hidden_type']):
  for control in ['known','background']:
   sub=part[part.actual_type!='background'] if control=='known' else part[(part.actual_type=='background')|part.is_hidden]
   if sub.is_hidden.nunique()==2:auc.append(dict(method=method,repeat=repeat,hidden_type=hidden,control=control,auc=roc_auc_score(sub.is_hidden,sub.score),n_hidden=int(sub.is_hidden.sum()),n_control=int((~sub.is_hidden).sum())))
 pd.DataFrame(auc).to_csv(OUT/'held_type_novelty_summary.csv',index=False)
 # Candidate descriptions use all known references descriptively, not for independent evaluation.
 feature_names=[f'{bp}bp_{name}' for bp in SCALES for name in FEATURE_NAMES]
 a,b=fit_route('localized_geometry',routes[1]['localized_geometry'],routes[2]['localized_geometry'],np.arange(nknown))
 reftrain=np.flatnonzero(known.split.eq('train').to_numpy());refcal=np.flatnonzero(known.split.ne('train').to_numpy())
 novelty_a,novelty_b=fit_route('localized_geometry',routes[1]['localized_geometry'],routes[2]['localized_geometry'],reftrain)
 calibration_dist=cdist(novelty_a[refcal],novelty_a[reftrain]).min(1)
 novelty_cutoff=float(np.quantile(calibration_dist,.95))
 candidate_reference_distance={r:cdist(x,novelty_a[reftrain]).min(1) for r,x in [(1,novelty_a),(2,novelty_b)]}
 scoretable=table.set_index(['item_index','repeat','bin_bp']);nearest=cdist(a[nknown:],a[:nknown]);cr=[]
 for row in candidate.itertuples():
  i=nknown+row.scan_index;idx=int(np.argmin(nearest[row.scan_index]));diff=a[i]-a[idx];top=np.argsort(-abs(diff))[:3]
  b1=arrays[1]['bounds'][i,1];b2=arrays[2]['bounds'][i,1]
  m=scoretable.loc[(i,1,100)];bgrows=table[(table.repeat==1)&(table.bin_bp==100)&(table.item_index>=nknown)]
  bgrows=bgrows[bgrows.item_index.isin(np.flatnonzero(trainbg)+nknown)]
  bgcenters=centers[bgrows.item_index.to_numpy()]
  gaps=abs(bgcenters-row.center)
  bgrows=bgrows[np.minimum(gaps,LENGTH-gaps)>=25600]
  controls=bgrows[(abs(bgrows.log_density-m.log_density)<=.5)&(abs(bgrows.local_support_fraction-m.local_support_fraction)<=.15)&(abs(bgrows.centroid_separation-m.centroid_separation)<=.1)]
  tail=(1+int((controls.localization_score>=m.localization_score).sum()))/(1+len(controls)) if len(controls) else None
  ratio=float(np.exp(m.log_density-scoretable.loc[(idx,1,100)].log_density))
  diff2=b[i]-b[idx]
  signs_agree=int((np.sign(diff[top])==np.sign(diff2[top])).sum())
  novel_reference=bool(candidate_reference_distance[1][i]>novelty_cutoff and candidate_reference_distance[2][i]>novelty_cutoff)
  differences='; '.join(f'{feature_names[j]}: candidate={routes[1]["localized_geometry"][i,j]:.3g}, nearest_known={routes[1]["localized_geometry"][idx,j]:.3g}, scaled_difference_rep1={diff[j]:+.2f}, scaled_difference_rep2={diff2[j]:+.2f}' for j in top)
  rr=stability[(stability.item_id==f'scan_{row.scan_index}')&(stability.bin_bp==100)].iloc[0]
  cr.append(dict(candidate_id=row.candidate_id,scan_index=row.scan_index,chrom=row.chrom,scan_start=row.start,scan_end=row.end,scan_length_bp=row.length_bp,
    estimated_start=int(b1[0]),estimated_end=int(b1[1]),estimated_center=int(b1.mean()),estimated_length_bp=int(b1[1]-b1[0]),
    rep2_estimated_start=int(b2[0]),rep2_estimated_end=int(b2[1]),repeat_boundary_iou=rr.repeat_boundary_iou,repeat_center_shift_bp=rr.repeat_center_shift_bp,
    aligned_shape_repeat_correlation=rr.aligned_image_repeat_correlation,whole_window_repeat_oe_correlation=row.repeat_oe_correlation,
    nearest_known_shape_id=known.structure_id.iloc[idx],nearest_known_shape_type=known.type.iloc[idx],shape_distance=float(nearest[row.scan_index,idx]),
    specific_geometry_distinction=differences,top3_geometry_difference_directions_agree=signs_agree,
    novelty_reference_distance_rep1=float(candidate_reference_distance[1][i]),novelty_reference_distance_rep2=float(candidate_reference_distance[2][i]),known_reference_cutoff95=novelty_cutoff,
    both_repeats_beyond_known_reference=novel_reference,comparison_heatmap_file=f'heatmaps/{row.candidate_id}_known_comparison.png',matched_background_count=len(controls),matched_background_localization_tail=tail,
    evidence_status='localized unannotated candidate; new morphology not established',heatmap_file=f'heatmaps/{row.candidate_id}.png'))
  fig,axes=plt.subplots(1,4,figsize=(13,3.7),constrained_layout=True)
  for ax,(source,repeat,title) in zip(axes,[(i,1,f'{row.candidate_id} rep1'),(i,2,f'{row.candidate_id} rep2'),(idx,1,f'{known.structure_id.iloc[idx]} rep1'),(idx,2,f'{known.structure_id.iloc[idx]} rep2')]):
   im=ax.imshow(np.ma.masked_where(~arrays[repeat]['masks'][source,1],arrays[repeat]['images'][source,1]),origin='lower',cmap='magma',vmin=0,vmax=2);ax.set(title=title,xlabel='Aligned footprint bin')
  fig.colorbar(im,ax=axes,label='Amplitude-normalized log(1+O/E)',shrink=.7)
  fig.savefig(OUT/f'heatmaps/{row.candidate_id}_known_comparison.png',dpi=130);plt.close(fig)
  # Three physical scales each shown at native spatial span, with independently localized boundaries.
  fig,axes=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
  exp=json.loads((OUT/'distance_expectation.json').read_text());pair_fields={}
  for ch,bp in enumerate(SCALES):
   for r in (1,2):
    if bp==50:matrix=np.load(f'data/processed/fine_scan_rep{r}.npz')['raw'][row.scan_index,0]
    else:matrix=np.load(f'data/processed/scan_rep{r}.npz')['raw'][row.scan_index,ch-1]
    coverage=np.load(f'data/processed/coverage_rep{r}.npz')['coverage'];support=support_at(coverage,np.array([row.center]),bp)[0]
    field,_,valid,oe=normalized_field(matrix,np.asarray(exp[str(bp)]),support,bp)
    start=int(np.floor((row.center+bp/2)/bp)-64)*bp
    if bp==100:pair_fields[r]=(oe,valid,start)
    ax=axes[r-1,ch];im=ax.imshow(np.ma.masked_where(~valid,field),origin='lower',cmap='magma',vmin=0,vmax=2,extent=[start/1000,(start+128*bp)/1000]*2)
    lo,hi=arrays[r]['bounds'][i,ch]/1000
    ax.add_patch(plt.Rectangle((lo,lo),hi-lo,hi-lo,fill=False,color='cyan',lw=1));ax.set(title=f'WT rep{r}, {bp} bp',xlabel='Genome coordinate (kb)')
  lo=int((b1[0]-pair_fields[1][2])//100);hi=int((b1[1]-pair_fields[1][2])//100)
  mask=pair_fields[1][1][lo:hi,lo:hi]&pair_fields[2][1][lo:hi,lo:hi]&np.triu(np.ones((hi-lo,hi-lo),bool),1)
  o1=pair_fields[1][0][lo:hi,lo:hi][mask];o2=pair_fields[2][0][lo:hi,lo:hi][mask]
  cr[-1]['same_coordinate_local_oe_correlation']=float(np.corrcoef(o1,o2)[0,1]) if len(o1)>3 and o1.std()>0 and o2.std()>0 else 0.
  cr[-1]['same_coordinate_local_valid_pairs']=int(len(o1))
  fig.colorbar(im,ax=axes,label='Amplitude-normalized log(1+O/E)',shrink=.7);fig.suptitle(f'{row.candidate_id}: estimated contact footprint (cyan); unconfirmed morphology')
  fig.savefig(OUT/f'heatmaps/{row.candidate_id}.png',dpi=130);plt.close(fig)
 result=pd.DataFrame(cr)
 # Fixed background-geometry radius for regrouping, without candidate-driven tuning.
 bgids=np.flatnonzero(trainbg)+nknown;eps=float(np.quantile(pdist(a[bgids]),.05))
 labels=DBSCAN(eps=eps,min_samples=2).fit_predict(a[nknown+candidate.scan_index.to_numpy()])
 result['cluster']=labels
 result['local_evidence_qualified']=(result.repeat_boundary_iou>=.5)&(result.aligned_shape_repeat_correlation>=.5)&(result.same_coordinate_local_oe_correlation>=.5)&(result.matched_background_count>=20)&(result.matched_background_localization_tail<=.05)&result.both_repeats_beyond_known_reference&(result.top3_geometry_difference_directions_agree>=2)
 result.to_csv(OUT/'localized_candidates.csv',index=False)
 groups=[]
 for cluster in sorted(set(labels)-{-1}):
  members=np.flatnonzero(labels==cluster);globalids=nknown+candidate.scan_index.to_numpy()[members]
  distances=cdist(b[globalids],b[globalids]);np.fill_diagonal(distances,0)
  groups.append(dict(cluster=cluster,members=len(members),candidate_ids=';'.join(candidate.candidate_id.iloc[members]),rep2_pairwise_max_distance=float(distances.max()),fixed_radius=eps,rep2_within_fixed_radius=bool(distances.max()<=eps),all_members_local_evidence_qualified=bool(result.local_evidence_qualified.iloc[members].all()),tentative_new_shape_cluster=bool(distances.max()<=eps and result.local_evidence_qualified.iloc[members].all())))
 pd.DataFrame(groups,columns=['cluster','members','candidate_ids','rep2_pairwise_max_distance','fixed_radius','rep2_within_fixed_radius','all_members_local_evidence_qualified','tentative_new_shape_cluster']).to_csv(OUT/'localized_clusters.csv',index=False)
 np.savez_compressed(OUT/'candidate_cluster_centers.npz',**{f'cluster{g}':a[nknown+candidate.scan_index.to_numpy()[labels==g]].mean(0) for g in set(labels) if g>=0})
 save_json(OUT/'summary.json',dict(known_structures=nknown,scan_windows=nscan,candidates=len(result),estimated_lengths_bp=sorted(result.estimated_length_bp.unique().astype(int).tolist()),
  median_candidate_repeat_boundary_iou=float(result.repeat_boundary_iou.median()),median_candidate_local_shape_repeat_correlation=float(result.aligned_shape_repeat_correlation.median()),
  candidate_clusters=len(groups),tentative_new_shape_clusters=sum(g['tentative_new_shape_cluster'] for g in groups),candidate_local_evidence_qualified=int(result.local_evidence_qualified.sum()),classification=metrics,known_normalization_leakage_note='Old globally fitted background expectation reused; fold geometry/PCA/classifiers refit locally; retrospective sensitivity evaluation'))
 print(pd.DataFrame(metrics).to_string(index=False));print(pd.DataFrame(auc).to_string(index=False));print(result[['candidate_id','estimated_length_bp','repeat_boundary_iou','nearest_known_shape_id','matched_background_count','matched_background_localization_tail']].to_string(index=False))

if __name__=='__main__':main()

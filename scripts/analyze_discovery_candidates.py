"""Analyze every retained scan candidate; cluster composition never gates output.

Known overlap, repeat quality, spatial redundancy and shape novelty are evidence
columns. A mixed cluster or singleton remains part of the analysis inventory.
"""
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.preprocessing import RobustScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dlpro.candidate_evidence import assess_inventory
from dlpro.local_morphology import normalized_field,FEATURE_NAMES,interval_iou
from dlpro.coverage_support import coarse_support
from dlpro.experiment_core import LENGTH,save_json

OUT=Path('results/discovery_candidate_analysis')
STATUS={
 'known_associated':'已知结构关联候选',
 'unannotated_known_like':'未注释且接近已知形态的候选',
 'unusual_shape':'形态异常待验证候选',
 'insufficient':'重复或信号质量不足的候选',
}

def genomic_segments(center,bp=100,lo=0,hi=128):
 total=(LENGTH+bp-1)//bp;first=int(np.floor((center+bp/2)/bp))-64
 bins=(first+np.arange(lo,hi))%total
 breaks=np.r_[0,np.flatnonzero(np.diff(bins)!=1)+1,len(bins)];parts=[]
 for a,b in zip(breaks[:-1],breaks[1:]):parts.append((int(bins[a]*bp),int(min((bins[b-1]+1)*bp,LENGTH))))
 return parts

def overlaps(parts,known,buffer=0):
 result=np.zeros(len(known),bool)
 for a,b in parts:
  for shift in [-LENGTH,0,LENGTH]:result|=(known.start.to_numpy()<b+buffer+shift)&(known.end.to_numpy()>a-buffer+shift)
 return result

def edge_distances(parts,known):
 gaps=np.full(len(known),np.inf)
 for a,b in parts:
  for shift in [-LENGTH,0,LENGTH]:gaps=np.minimum(gaps,np.maximum(np.maximum(known.start.to_numpy()-(b+shift),(a+shift)-known.end.to_numpy()),0))
 return gaps.astype(int)

def main():
 OUT.mkdir(exist_ok=True);(OUT/'heatmaps').mkdir(exist_ok=True)
 candidates=pd.read_csv('results/discovery/candidates.csv');known=pd.read_csv('data/processed/known_windows.csv');scan=pd.read_csv('data/processed/scan_windows.csv')
 assert len(candidates)==294 and len(known)==344
 n=len(known);centers=scan.center.to_numpy()
 measurement=np.load('results/local_contact_morphology/measurements.npz')
 geometry={r:measurement[f'fixed_geometry_rep{r}'].reshape(n+len(scan),-1) for r in (1,2)}
 scaler=RobustScaler(quantile_range=(10,90)).fit(geometry[1][:n])
 z={r:scaler.transform(geometry[r]) for r in (1,2)}
 nearest=cdist(z[1][n:],z[1][:n]).argmin(1)
 training=np.flatnonzero(known.split.eq('train'));calibration=np.flatnonzero(known.split.ne('train'))
 calibration_scaler=RobustScaler(quantile_range=(10,90)).fit(geometry[1][training])
 cz={r:calibration_scaler.transform(geometry[r]) for r in (1,2)}
 calibration_dist=cdist(cz[1][calibration],cz[1][training]).min(1)
 cutoff=float(np.quantile(calibration_dist,.95))
 novelty={r:cdist(cz[r][n:],cz[1][training]).min(1) for r in (1,2)}
 raw={r:np.load(f'data/processed/scan_rep{r}.npz')['raw'][:,0] for r in (1,2)}
 knownraw={r:np.load(f'data/processed/known_rep{r}.npz')['raw'][:,0] for r in (1,2)}
 support={r:coarse_support(np.load(f'data/processed/coverage_rep{r}.npz')['coverage'],centers)[:,0]>=.8 for r in (1,2)}
 ksupport={r:coarse_support(np.load(f'data/processed/coverage_rep{r}.npz')['coverage'],known.center.to_numpy())[:,0]>=.8 for r in (1,2)}
 expected=np.asarray(json.loads(Path('results/local_contact_morphology/distance_expectation.json').read_text())['100'])
 d=abs(np.arange(128)[:,None]-np.arange(128)[None,:]);triangle=np.triu(np.ones((128,128),bool),2)
 table=pd.read_csv('results/local_contact_morphology/per_window_geometry.csv')
 primary=table[(table.repeat==1)&(table.bin_bp==100)].set_index('item_index')
 bgids=np.flatnonzero(scan.training_background.to_numpy(bool))+n
 bg=primary.loc[bgids].copy();bgcenters=centers[bgids-n]
 cross_scale=pd.read_csv('results/local_contact_morphology/cross_scale_localization.csv')
 coords=[];rows=[];names=[f'{bp}bp_{name}' for bp in [50,100,200] for name in FEATURE_NAMES]
 for candidate in candidates.itertuples():
  i=int(candidate.scan_index);index=n+i;j=int(nearest[i]);parts=genomic_segments(int(candidate.center))
  direct=overlaps(parts,known);buffered=overlaps(parts,known,200)
  # Context comes from actual 200 bp source bins, including the short chromosome-end bin.
  context=genomic_segments(int(candidate.center),200);full=overlaps(context,known,200)
  gaps=edge_distances(parts,known);kcoord=int(np.argmin(gaps))
  valid=triangle&support[1][i,:,None]&support[1][i,None,:]&support[2][i,:,None]&support[2][i,None,:]
  oe1=(raw[1][i]/expected[d])[valid];oe2=(raw[2][i]/expected[d])[valid]
  correlation=float(np.corrcoef(oe1,oe2)[0,1]) if len(oe1)>3 and oe1.std()>0 and oe2.std()>0 else None
  fraction=float(valid.sum()/triangle.sum());quality=bool(correlation is not None and correlation>=.5 and fraction>=.5)
  difference=z[1][index]-z[1][j];difference2=z[2][index]-z[2][j];top=np.argsort(-abs(difference))[:3]
  specific='; '.join(f'{names[c]}: candidate={geometry[1][index,c]:.3g}, reference={geometry[1][j,c]:.3g}, scaled_delta_rep1={difference[c]:+.2f}, scaled_delta_rep2={difference2[c]:+.2f}' for c in top)
  unusual=bool(novelty[1][i]>cutoff and novelty[2][i]>cutoff)
  if not quality:status='insufficient'
  elif buffered.any():status='known_associated'
  elif unusual:status='unusual_shape'
  else:status='unannotated_known_like'
  m=primary.loc[index];dist=abs(bgcenters-candidate.center)
  match=(np.minimum(dist,LENGTH-dist)>=25600)&(abs(bg.log_density.to_numpy()-m.log_density)<=.5)&(abs(bg.local_support_fraction.to_numpy()-m.local_support_fraction)<=.15)&(abs(bg.centroid_separation.to_numpy()-m.centroid_separation)<=.1)
  controls=bg[match];tail=float((1+(controls.localization_score>=m.localization_score).sum())/(1+len(controls))) if len(controls) else None
  b1=measurement['bounds_rep1'][index,1];b2=measurement['bounds_rep2'][index,1]
  windowstart=int(np.floor((candidate.center+50)/100)-64)*100
  lo,hi=((b1-windowstart)//100).astype(int);localparts=genomic_segments(int(candidate.center),100,int(lo),int(hi))
  difflabels=int((np.sign(difference[top])==np.sign(difference2[top])).sum())
  cs=cross_scale[cross_scale.item_id==f'scan_{i}']
  note=f'扫描窗口直接重叠{int(direct.sum())}条已知注释，200 bp缓冲范围内{int(buffered.sum())}条，完整上下文内{int(full.sum())}条；整窗几何最近参考为{known.structure_id.iloc[j]}（{known.type.iloc[j]}）；三项最大几何差异中{difflabels}项跨重复同向。'
  row=dict(candidate_id=candidate.candidate_id,scan_index=i,chrom=candidate.chrom,start=parts[0][0],end=parts[-1][1],center=int(candidate.center),length_bp=sum(b-a for a,b in parts),wraps_origin=len(parts)>1,
   genomic_segments=json.dumps(parts),cluster=int(candidate.cluster),shape_score=float(candidate.shape_score),rep2_shape_score=float(candidate.rep2_shape_score),
   repeat_oe_correlation=correlation,original_repeat_oe_correlation=float(candidate.repeat_oe_correlation),common_supported_triangle_fraction=fraction,quality_pass=quality,
   exact_known_overlap=bool(direct.any()),known_ids=';'.join(known.structure_id[direct]),known_buffer_overlap=bool(buffered.any()),known_full_context_overlap=bool(full.any()),
   nearest_known_coordinate_id=known.structure_id.iloc[kcoord],nearest_known_edge_gap_bp=int(gaps[kcoord]),
   nearest_known_shape_id=known.structure_id.iloc[j],nearest_known_shape_type=known.type.iloc[j],nearest_known_shape_distance=float(np.linalg.norm(z[1][index]-z[1][j])),
   known_structure_distinction=note,specific_geometry_distinction=specific,top3_geometry_difference_directions_agree=difflabels,
   novelty_distance_rep1=float(novelty[1][i]),novelty_distance_rep2=float(novelty[2][i]),known_reference_distance95=cutoff,both_repeats_beyond_known_distance=unusual,
   estimated_start=localparts[0][0],estimated_end=localparts[-1][1],estimated_length_bp=sum(b-a for a,b in localparts),estimated_genomic_segments=json.dumps(localparts),
   local_repeat_boundary_iou=interval_iou(b1,b2),minimum_cross_scale_interval_iou=float(cs.physical_interval_iou.min()),
   matched_background_count=len(controls),matched_background_localization_tail=tail,
   evidence_category=status,evidence_category_zh=STATUS[status],tentative_new_shape=False,
   heatmap_file=f'heatmaps/{candidate.candidate_id}.png',all_candidates_retained=True)
  rows.append(row)
  if quality:coords.append((candidate.candidate_id,int(candidate.center)))
  fig,axes=plt.subplots(1,4,figsize=(12,3.5),constrained_layout=True)
  for panel,(r,reference) in enumerate([(1,False),(2,False),(1,True),(2,True)]):
   matrix=knownraw[r][j] if reference else raw[r][i]
   supported=ksupport[r][j] if reference else support[r][i]
   field,_,mask,_=normalized_field(matrix,expected,supported,100)
   im=axes[panel].imshow(np.ma.masked_where(~mask,field),origin='lower',cmap='magma',vmin=0,vmax=2)
   label=known.structure_id.iloc[j] if reference else candidate.candidate_id
   axes[panel].set(title=f'{label}, WT rep{r}',xlabel='Relative 100 bp bin')
   if not reference:
    bx=measurement[f'bounds_rep{r}'][index,1];a,b=((bx-windowstart)/100)
    axes[panel].add_patch(plt.Rectangle((a,a),b-a,b-a,fill=False,color='cyan',lw=.7))
  fig.colorbar(im,ax=axes,label='Amplitude-normalized log(1+O/E)',shrink=.8)
  fig.suptitle(f'Cluster {candidate.cluster}; {status}; supported O/E r={correlation:.3f}' if correlation is not None else f'Cluster {candidate.cluster}; unavailable repeat correlation',fontsize=10)
  fig.savefig(OUT/row['heatmap_file'],dpi=100);plt.close(fig)
  if len(rows)%50==0:print('analyzed',len(rows),'/',len(candidates),flush=True)
 result=pd.DataFrame(rows)
 # Keep redundancy visible; representative status never removes a row.
 representatives=[]
 for row in result[result.quality_pass].sort_values(['shape_score','candidate_id'],ascending=[False,True]).itertuples():
  distances=abs(np.asarray([x[1] for x in representatives])-row.center)
  if np.all(np.minimum(distances,LENGTH-distances)>=25600):representatives.append((row.candidate_id,row.center))
 result['spatial_representative']=result.candidate_id.isin([x[0] for x in representatives])
 result['spatial_reference_candidate']=''
 for row in result[result.quality_pass].itertuples():
  distances=np.minimum(abs(np.asarray([x[1] for x in representatives])-row.center),LENGTH-abs(np.asarray([x[1] for x in representatives])-row.center))
  result.loc[result.candidate_id==row.candidate_id,'spatial_reference_candidate']=representatives[int(np.argmin(distances))][0]
 result=assess_inventory(result)
 result.to_csv(OUT/'candidate_inventory.csv',index=False)
 result[result.quality_pass].to_csv(OUT/'quality_supported_candidates.csv',index=False)
 result[result.quality_pass&~result.known_buffer_overlap].to_csv(OUT/'unannotated_candidates.csv',index=False)
 cluster_rows=[]
 for group,part in result.groupby('cluster'):
  adequate=part[part.quality_pass];known_members=int(part.known_buffer_overlap.sum());unannotated=int((~part.known_buffer_overlap).sum())
  cluster_rows.append(dict(cluster=int(group),members=len(part),quality_supported=len(adequate),known_associated_members=known_members,unannotated_members=unannotated,
   composition='mixed' if known_members and unannotated else ('known_associated' if known_members else 'unannotated'),
   quality_unannotated_members=int((~adequate.known_buffer_overlap).sum()),spatial_representatives=int(part.spatial_representative.sum()),
   representative_candidate=part.sort_values('shape_score',ascending=False).candidate_id.iloc[0],member_ids=';'.join(part.candidate_id),
   analysis_eligible=True,minimum_members_gate_used=False,known_member_veto_used=False,tentative_new_shape_members=int(part.tentative_new_shape.sum())))
 pd.DataFrame(cluster_rows).to_csv(OUT/'cluster_inventory.csv',index=False)
 subgroups=[]
 for (cluster,kind),part in result[result.quality_pass].groupby(['cluster','nearest_known_shape_type']):
  subgroups.append(dict(cluster=int(cluster),nearest_known_shape_type=kind,members=len(part),known_associated=int(part.known_buffer_overlap.sum()),unannotated=int((~part.known_buffer_overlap).sum()),member_ids=';'.join(part.candidate_id),interpretation='Reference-resemblance subgroup; not a newly discovered biological class'))
 pd.DataFrame(subgroups).to_csv(OUT/'mixed_cluster_reference_subgroups.csv',index=False)
 embedding=np.load('results/discovery/embedding.npz');pd.DataFrame(embedding['centers'],columns=[f'PC{i+1}' for i in range(embedding['centers'].shape[1])]).rename_axis('cluster').to_csv(OUT/'cluster_centers.csv')
 compressed=np.einsum('nd,kd->nk',embedding['latent']-embedding['pca_mean'],embedding['pca_components'],optimize=False)
 fig,ax=plt.subplots(figsize=(8,5));idx=result.scan_index.to_numpy()
 for status in STATUS:
  mask=result.evidence_category.eq(status).to_numpy();ax.scatter(compressed[idx[mask],0],compressed[idx[mask],1],s=20,label=f'{status} ({mask.sum()})',alpha=.8)
 ax.set(xlabel='Original shape PC1',ylabel='Original shape PC2',title='Candidate evidence categories; all retained');ax.legend(fontsize=8);fig.tight_layout();fig.savefig(OUT/'candidate_evidence_map.png',dpi=140);plt.close(fig)
 summary=dict(candidates=len(result),clusters=len(cluster_rows),mixed_clusters=sum(g['composition']=='mixed' for g in cluster_rows),quality_supported=int(result.quality_pass.sum()),
  evidence_counts={str(k):int(v) for k,v in result.evidence_category.value_counts().items()},quality_unannotated=int((result.quality_pass&~result.known_buffer_overlap).sum()),
  spatial_representatives=int(result.spatial_representative.sum()),tentative_new_shapes=int(result.tentative_new_shape.sum()),source='Frozen original 294 scan candidates and six AE/PCA clusters; candidate acquisition unchanged')
 save_json(OUT/'summary.json',summary)
 save_json(OUT/'protocol.json',dict(all_scan_candidates_retained=True,minimum_cluster_size_for_analysis=None,known_members_veto=False,
  candidate_source='results/discovery/candidates.csv',cluster_source='results/discovery/embedding.npz',quality_rule='common native-support upper triangle >=0.5 and supported O/E Pearson >=0.5',
  native_row_support=.8,distance_exclusion_bp=200,known_buffer_bp=200,spatial_representative_spacing_bp=25600,
  shape_reference='Whole-window geometric descriptors (three scales); descriptive nearest known reference, not a calibrated type prediction',
  novelty='Distance95 is an evidence flag only, never a row/cluster exclusion; no candidate automatically called novel',
  backgrounds='Same local-search outputs for density/support/diagonal-offset matched backgrounds; >=25.6kb from query; descriptive, not end-to-end selection-adjusted p values',
  evidence_categories=STATUS,heatmap={'bin_bp':100,'vmin':0,'vmax':2,'transform':'amplitude-normalized log(1+O/E)','panels':['candidate rep1','candidate rep2','nearest known geometry rep1','nearest known geometry rep2']},
  coordinate_system='0-based half-open; wrapped intervals given by genomic_segments; terminal chromosome bin has actual 52bp width',
  limitations=['Retrospective same-study analysis','Shape resemblance is not reliable subtype classification','Mixed-cluster subgroups defined by known-reference similarity, not new biological labels','Local bounds can refer to different objects at different scales','No biological new class established']))
 print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()

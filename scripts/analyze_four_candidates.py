"""Fixed four-candidate descriptive analysis; no new selection or significance claims."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dlpro.experiment_core import LENGTH,ShapeTransform,save_json
from dlpro.coverage_support import coarse_support
out=Path('results/four_candidate_analysis');out.mkdir(exist_ok=True)
f=pd.read_csv('results/candidate_context/eligibility.csv');chosen=f[f.independent_context_member];assert chosen.candidate_id.tolist()==['U001','U004','U005','U007']
all_windows=pd.read_csv('results/candidate_conditions/unannotated_consensus.csv');known=pd.read_csv('data/annotations/structures.csv');samples=pd.read_csv('data/annotations/samples.csv');samples=samples[samples.condition.isin(['37C','HNS_mut','FIS_mut'])].sort_values(['condition','replicate']);raw=np.load('results/candidate_conditions/windows.npz');distance=abs(np.arange(128)[:,None]-np.arange(128)[None,:]);expected=ShapeTransform.load('results/discovery_spatial_cv/fold0/transform.npz').expected;rows=[];nearest=[]
coverage=[np.load(f'data/processed/coverage_rep{r}.npz')['coverage'] for r in [1,2]];support=[coarse_support(c,all_windows.center.to_numpy()) for c in coverage]
for candidate in chosen.itertuples():
 pos=all_windows.index[all_windows.candidate_id==candidate.candidate_id][0];center=int(candidate.center)
 distances=[]
 for annotation in known.itertuples():
  gap=min(max(annotation.start-(center+12800+shift),(center-12800+shift)-annotation.end,0) for shift in [-LENGTH,0,LENGTH]);distances.append(gap)
 ids=np.argsort(distances)[:3]
 for rank,i in enumerate(ids):
  a=known.iloc[i];nearest.append(dict(candidate_id=candidate.candidate_id,rank=rank+1,structure_id=a.structure_id,type=a.type,annotation_start=int(a.start),annotation_end=int(a.end),context_edge_gap_bp=int(distances[i]),center_distance_bp=int(min(abs(center-a.center),LENGTH-abs(center-a.center)))))
 fig,axes=plt.subplots(2,6,figsize=(16,6),constrained_layout=True)
 for channel,bp in enumerate([100,200]):
  fields=[]
  supported_rows=(support[0][pos,channel]>=.8)&(support[1][pos,channel]>=.8)
  for source in samples.itertuples():supported_rows&=raw[source.sample_id][pos,channel].sum(-1)>0
  supported=supported_rows[:,None]&supported_rows[None,:]
  for sample in samples.itertuples():
   a=raw[sample.sample_id][pos,channel];oe=a/expected[channel,distance];valid=distance>=2;amp=oe[valid].mean();shape=np.log1p(oe/max(amp,1e-12));fields.append(shape)
   for region,mask in [('central',(np.indices((128,128))[0]>=48)&(np.indices((128,128))[0]<80)&(np.indices((128,128))[1]>=48)&(np.indices((128,128))[1]<80)),('context',np.ones((128,128),bool))]:
    mask=mask&valid
    for suffix,eligible in [('',mask),('_supported',mask&supported)]:
     rows.append(dict(candidate_id=candidate.candidate_id,center=center,bin_bp=bp,sample_id=sample.sample_id,condition=sample.condition,replicate=int(sample.replicate),region=region+suffix,supported_fraction=float(eligible.sum()/max(mask.sum(),1)),mean_density=float(a[eligible].mean()),mean_oe=float(oe[eligible].mean()),normalized_shape_mean=float(shape[eligible].mean())))
  vmax=np.quantile(np.stack(fields),.995)
  for col,(sample,field) in enumerate(zip(samples.itertuples(),fields)):
   ax=axes[channel,col];im=ax.imshow(field,origin='lower',cmap='magma',vmin=0,vmax=vmax);ax.set_title(f'{sample.condition} r{int(sample.replicate)}',fontsize=9);ax.set_xlabel(f'{bp}bp bins');ax.add_patch(plt.Rectangle((48,48),32,32,fill=False,color='cyan',lw=.7))
  fig.colorbar(im,ax=axes[channel,:],shrink=.8,label='Amplitude-normalized log(1+O/E)')
 fig.suptitle(f'{candidate.candidate_id}: center {center:,}; exploratory, not a new class');fig.savefig(out/f'{candidate.candidate_id}.png',dpi=130);plt.close(fig)
measure=pd.DataFrame(rows);measure.to_csv(out/'region_measurements.csv',index=False);pd.DataFrame(nearest).to_csv(out/'nearest_annotations.csv',index=False)
contrasts=[]
for (candidate,bp,region),g in measure.groupby(['candidate_id','bin_bp','region']):
 wt=g[g.condition=='37C'].mean_density.to_numpy()
 for condition in ['HNS_mut','FIS_mut']:
  values=g[g.condition==condition].mean_density.to_numpy();ratios=values[:,None]/wt[None,:]
  contrasts.append(dict(candidate_id=candidate,bin_bp=bp,region=region,condition=condition,mean_ratio=float(values.mean()/wt.mean()),all_pair_ratio_min=float(ratios.min()),all_pair_ratio_max=float(ratios.max()),both_repeats_below_all_wt=bool(values.max()<wt.min()),both_repeats_above_all_wt=bool(values.min()>wt.max())))
pd.DataFrame(contrasts).to_csv(out/'condition_contrasts.csv',index=False)
relative=measure.pivot(index=['candidate_id','bin_bp','sample_id','condition','replicate'],columns='region',values='mean_density').reset_index();relative['central_to_context']=relative.central/relative.context;relative['supported_central_to_context']=relative.central_supported/relative.context_supported;relative.to_csv(out/'central_to_context.csv',index=False)
relative_rows=[]
for (candidate,bp),g in relative.groupby(['candidate_id','bin_bp']):
 wt=g[g.condition=='37C'].supported_central_to_context.to_numpy()
 for condition in ['HNS_mut','FIS_mut']:
  values=g[g.condition==condition].supported_central_to_context.to_numpy();ratios=values[:,None]/wt[None,:];relative_rows.append(dict(candidate_id=candidate,bin_bp=bp,condition=condition,supported_central_enrichment_ratio=float(values.mean()/wt.mean()),cross_repeat_min=float(ratios.min()),cross_repeat_max=float(ratios.max())))
pd.DataFrame(relative_rows).to_csv(out/'relative_enrichment_contrasts.csv',index=False)

quality=measure[(measure.region=='central_supported')&(measure.condition=='37C')&(measure.replicate==1)][['candidate_id','bin_bp','supported_fraction']].copy();quality['central_support_below_half']=quality.supported_fraction<.5;quality.to_csv(out/'central_support_flags.csv',index=False)
comparisons=pd.read_csv('results/candidate_conditions/condition_comparisons.csv');comparisons[comparisons.candidate_id.isin(chosen.candidate_id)].to_csv(out/'repeat_shape_comparisons.csv',index=False)
save_json(out/'protocol.json',dict(candidates=chosen.candidate_id.tolist(),fixed_before_analysis=True,central_region='32x32 pixels centered in each128x128 image; physical widths3200/6400bp',context_widths_bp=[12800,25600],normalization='Per-million library count density; WT train fold0 expectation shared; amplitude normalization for heatmap only',distance='Circular interval edge gap, complete25.6kb candidate context to annotation; no extra buffer in reported distance',limitations=['Selection exploratory, no independent confirmatory test','Two repeats per condition; four cross-repeat ratios are not four independent biological samples','Ratios descriptive, not significance or causal protein effects','Central rectangle is a fixed geometric summary, not a learned structural boundary','Heatmaps show unmasked pixels; supported summaries use identical WT80%native-support plus all-six-sample nonzero-row mask; center support retained explicitly, particularly important forU005']))
print(pd.DataFrame(nearest).query('rank==1').to_string(index=False));print(pd.DataFrame(contrasts).query('bin_bp==100 and region=="central"').to_string(index=False))

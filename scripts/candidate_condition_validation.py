"""Exploratory consensus windows checked in WT, HNS_mut and FIS_mut repeats."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dlpro.experiment_core import ContactReader,ShapeTransform,LENGTH,hits,centered_indices,save_json

out=Path('results/candidate_conditions');out.mkdir(exist_ok=True);(out/'heatmaps').mkdir(exist_ok=True)
all_candidates=pd.read_csv('results/discovery_spatial_cv/candidates.csv');selected=all_candidates[(all_candidates.method=='morphology')&(all_candidates.budget==64)].groupby(['scan_index','center']).fold.nunique().reset_index(name='selected_folds');selected=selected[selected.selected_folds>=3].sort_values('center')
known=pd.read_csv('data/annotations/structures.csv');selected['known_overlap']=[bool(hits(int(c),12800,known,buffer=200).any()) for c in selected.center];unannotated=selected[~selected.known_overlap].copy();unannotated['candidate_id']=[f'U{i+1:03d}' for i in range(len(unannotated))];selected.to_csv(out/'consensus.csv',index=False);unannotated.to_csv(out/'unannotated_consensus.csv',index=False)
samples=pd.read_csv('data/annotations/samples.csv');chosen=samples[samples.condition.isin(['37C','HNS_mut','FIS_mut'])].sort_values(['condition','replicate']);sources={r.sample_id:r.sha256 for r in chosen.itertuples()};cache=out/'windows.npz';manifest=out/'sources.json';matrices={}
if cache.exists():
    assert json.loads(manifest.read_text())==sources;obj=np.load(cache);matrices={k:obj[k] for k in obj.files}
else:
    for sample in chosen.itertuples():
        reader=ContactReader(sample.path);values=[]
        for row in unannotated.itertuples():
            starts=centered_indices(row.center);values.append(np.stack([reader.window(starts[i],128,bp) for i,bp in enumerate([100,200])]))
        matrices[sample.sample_id]=np.asarray(values,dtype='float32');print('candidate conditions',sample.sample_id,len(values),flush=True)
    np.savez_compressed(cache,**matrices);save_json(manifest,sources)
tf=ShapeTransform.load('results/discovery_spatial_cv/fold0/transform.npz');d=abs(np.arange(128)[:,None]-np.arange(128)[None,:]);triangle=np.triu_indices(128,2);expected=tf.expected[0,d];rows=[];metrics=[]
for pos,row in enumerate(unannotated.itertuples()):
    shape={};signal={}
    for sample in chosen.itertuples():
        oe=matrices[sample.sample_id][pos,0]/expected;vector=oe[triangle];signal[sample.sample_id]=float(matrices[sample.sample_id][pos,0][triangle].mean());amp=max(float(vector.mean()),1e-8);shape[sample.sample_id]=np.log1p(oe/amp)
        rows.append(dict(candidate_id=row.candidate_id,center=int(row.center),selected_folds=int(row.selected_folds),sample_id=sample.sample_id,condition=sample.condition,replicate=int(sample.replicate),mean_off_diagonal_density=signal[sample.sample_id]))
    wt_ids=chosen[chosen.condition.eq('37C')].sample_id.tolist();wt_mean=(shape[wt_ids[0]]+shape[wt_ids[1]])/2;wt_signal=np.mean([signal[v] for v in wt_ids]);rep_correlations={}
    for condition in ['37C','HNS_mut','FIS_mut']:
        ids=chosen[chosen.condition.eq(condition)].sample_id.tolist();vectors=[shape[v][triangle] for v in ids];r=float(np.corrcoef(*vectors)[0,1]) if all(v.std()>0 for v in vectors) else None;rep_correlations[condition]=r;mean=(shape[ids[0]]+shape[ids[1]])/2;cross=float(np.corrcoef(wt_mean[triangle],mean[triangle])[0,1]) if mean[triangle].std()>0 and wt_mean[triangle].std()>0 else None
        metrics.append(dict(candidate_id=row.candidate_id,center=int(row.center),condition=condition,repeat_shape_correlation=r,wt_shape_correlation=cross,shape_rmse_vs_wt=float(np.sqrt(((mean-wt_mean)[triangle]**2).mean())),signal_ratio_vs_wt=float(np.mean([signal[v] for v in ids])/max(wt_signal,1e-12))))
    fig,axes=plt.subplots(2,3,figsize=(10,6),constrained_layout=True);vmax=np.quantile(np.stack(list(shape.values())),.995)
    for column,condition in enumerate(['37C','HNS_mut','FIS_mut']):
        for repeat in [1,2]:
            sample=chosen[(chosen.condition==condition)&(chosen.replicate==repeat)].iloc[0];ax=axes[repeat-1,column];im=ax.imshow(shape[sample.sample_id],vmin=0,vmax=vmax,cmap='magma',origin='lower');ax.set_title(f'{condition} rep{repeat}',fontsize=9)
    fig.suptitle(f'{row.candidate_id}; center {row.center:,}; selected in {row.selected_folds}/5 fits');fig.colorbar(im,ax=axes,label='Amplitude-normalized log(1+O/E)',shrink=.7);fig.savefig(out/'heatmaps'/f'{row.candidate_id}.png',dpi=120);plt.close(fig)
pd.DataFrame(rows).to_csv(out/'sample_measurements.csv',index=False);result=pd.DataFrame(metrics);result.to_csv(out/'condition_comparisons.csv',index=False)
wt=result[result.condition=='37C'];save_json(out/'protocol.json',dict(consensus_threshold_folds=3,primary_budget_per_fit=64,consensus_windows=len(selected),unannotated_consensus_windows=len(unannotated),wt_reproducible_windows=int((wt.repeat_shape_correlation>=.5).sum()),conditions=['37C','HNS_mut','FIS_mut'],repeats_per_condition=2,window_width_bp=12800,oe_distance_expectation='WT training backgrounds of outer fold0; shared reference across conditions',signal='Per-million library pair off-diagonal100bp pixel density; no ICE balancing',selection='Consensus of five spatial fits; exclude all known annotations plus200bp; mutants never used for selection',limitations=['Exploratory consensus aggregates fits that can see a location in training or validation; not out-of-fold novel discovery','Only two biological repeats per condition; descriptive differences, no causal inference or significance claims','Shared WT expectation and per-window amplitude normalization retain shape but may conceal overall condition intensity changes','No qualifying new-type cluster established; unannotated consensus does not confirm novel biology']))
print('consensus',len(selected),'unannotated',len(unannotated),'WT repeated',int((wt.repeat_shape_correlation>=.5).sum()),flush=True)

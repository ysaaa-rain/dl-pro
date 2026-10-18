"""Explore repeat-consistent changes in annotated structures across conditions."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dlpro.experiment_core import ContactReader, save_json
from dlpro.differential import local_enrichment, repeat_contrast

OUT=Path('results/structure_conditions'); OUT.mkdir(parents=True, exist_ok=True)
CACHE=Path('data/processed/structure_conditions'); CACHE.mkdir(parents=True, exist_ok=True)
f=pd.read_csv('data/annotations/structures.csv')
samples=pd.read_csv('data/annotations/samples.csv')
samples=samples[samples.condition.isin(['37C','HNS_mut','FIS_mut'])].sort_values(['condition','replicate'])
assert len(samples)==6 and len(f)==344
save_json(OUT/'protocol.json', dict(conditions=['37C','HNS_mut','FIS_mut'], structures=344, context='annotation plus one annotation-length flank on each side; bounds aligned to 200bp', resolutions_bp=[100,200], excluded_separation_bp=200, normalization='per-million library pairs; target versus distance-matched within-flank contact density', effect_threshold_log2=.5, minimum_cross_repeat_effect_log2=.25, sensitivity_threshold_log2=.25, replicate_design='unpaired; all four mutant-WT contrasts', repeat_noise_quantile=.95, inference='exploratory; two biological replicates do not support individual-structure significance testing', background_limitation='flanks may contain other known or unknown structures', source_samples=samples[['sample_id','path','sha256','library_pairs']].to_dict('records')))
matrices={}
for sample in samples.itertuples():
    path=CACHE/(sample.sample_id+'.npz')
    if not path.exists():
        reader=ContactReader(sample.path); arrays={}
        for j,r in enumerate(f.itertuples()):
            length=r.end-r.start; lo=((r.start-length)//200)*200; hi=int(np.ceil((r.end+length)/200))*200
            arrays[r.structure_id]=reader.window(lo//100,(hi-lo)//100,100)
            if j%100==0: print(sample.sample_id,j,flush=True)
        np.savez_compressed(path,**arrays)
    matrices[sample.sample_id]=np.load(path)
rows=[]
for r in f.itertuples():
    length=r.end-r.start; lo=((r.start-length)//200)*200
    metrics={}
    for sample in samples.itertuples():
        a=matrices[sample.sample_id][r.structure_id]
        for bp in [100,200]:
            b=a if bp==100 else a.reshape(len(a)//2,2,len(a)//2,2).mean(axis=(1,3))
            centers=lo+(np.arange(len(b))+.5)*bp
            metrics[(sample.condition,int(sample.replicate),bp,False)]=local_enrichment(b,centers,r.start,r.end,bp)
            metrics[(sample.condition,int(sample.replicate),bp,True)]=local_enrichment(b,centers,r.start,r.end,bp,True)
    for condition in ['HNS_mut','FIS_mut']:
        row=dict(structure_id=r.structure_id,type=r.type,chrom=r.chrom,start=r.start,end=r.end,condition=condition)
        for bp,correct,label in [(100,False,''),(200,False,'coarse_'),(100,True,'row_corrected_')]:
            wt=np.array([metrics[('37C',rep,bp,correct)][0] for rep in [1,2]])
            mut=np.array([metrics[(condition,rep,bp,correct)][0] for rep in [1,2]])
            effect,consistent,minimum,wt_noise,mut_noise=repeat_contrast(wt,mut)
            row.update({label+'log2_enrichment_change':effect,label+'all_cross_direction_consistent':consistent,label+'minimum_cross_effect':minimum,label+'wt_repeat_difference':wt_noise,label+'mutant_repeat_difference':mut_noise})
            if not label:
                for rep in [1,2]:
                    row[f'WT_rep{rep}_enrichment']=wt[rep-1];row[f'mutant_rep{rep}_enrichment']=mut[rep-1]
                row['core_pairs']=metrics[('37C',1,bp,correct)][2]
                row['log2_absolute_signal_change']=float(np.log2(np.mean([metrics[(condition,rep,bp,correct)][1] for rep in [1,2]])/np.mean([metrics[('37C',rep,bp,correct)][1] for rep in [1,2]])))
        rows.append(row)
g=pd.DataFrame(rows)
summaries=[]
for condition in ['HNS_mut','FIS_mut']:
    ix=g.condition==condition; s=g.loc[ix]
    cutoff=float(np.quantile(np.r_[s.wt_repeat_difference,s.mutant_repeat_difference],.95))
    effect=s.log2_enrichment_change
    qualified=(effect.abs()>=max(.5,cutoff)) & s.all_cross_direction_consistent & (s.minimum_cross_effect>=.25) & (np.sign(effect)==np.sign(s.coarse_log2_enrichment_change)) & (s.coarse_log2_enrichment_change.abs()>=.25) & s.coarse_all_cross_direction_consistent & (np.sign(effect)==np.sign(s.row_corrected_log2_enrichment_change)) & (s.row_corrected_log2_enrichment_change.abs()>=.25) & (s.core_pairs>=20)
    g.loc[ix,'repeat_noise_cutoff_log2']=cutoff;g.loc[ix,'evidence_qualified']=qualified
    summaries.append(dict(condition=condition,structures=len(s),repeat_noise_cutoff_log2=cutoff,qualified=int(qualified.sum()),weakened=int((qualified & (effect<0)).sum()),strengthened=int((qualified & (effect>0)).sum()),median_absolute_log2_change=float(s.log2_absolute_signal_change.median()),median_local_log2_change=float(effect.median())))
g.to_csv(OUT/'structure_changes.csv',index=False)
selected=g[g.evidence_qualified.astype(bool)].copy();selected['effect_magnitude']=selected.log2_enrichment_change.abs();selected=selected.sort_values('effect_magnitude',ascending=False);selected.to_csv(OUT/'repeat_consistent_changes.csv',index=False)
save_json(OUT/'summary.json', dict(contrasts=summaries,finite_effects=bool(np.isfinite(g.log2_enrichment_change).all()),limitations=['Retrospective exploratory comparison, not independent confirmation.','Two unpaired biological repeats per condition; no per-structure p-values or FDR claims.','Shared/overlapping structures are not independent observations.','Flank enrichment measures local contrast; it is not proof of loss or gain of a structural class.','Row-marginal correction is a sensitivity analysis and can remove biological signal.']))
fig,axes=plt.subplots(1,2,figsize=(11,4))
for ax,condition in zip(axes,['HNS_mut','FIS_mut']):
    s=g[g.condition==condition]
    ax.scatter(s.log2_absolute_signal_change,s.log2_enrichment_change,c=np.where(s.evidence_qualified,'#d44','#999'),s=15,alpha=.7)
    ax.axhline(0,color='k',lw=.5);ax.axvline(0,color='k',lw=.5);ax.set(title=condition,xlabel='Absolute signal log2 change',ylabel='Local enrichment log2 change')
fig.tight_layout();fig.savefig(OUT/'absolute_and_local_changes.png',dpi=160);plt.close(fig)
for condition in ['HNS_mut','FIS_mut']:
    for chosen in selected[selected.condition==condition].head(4).itertuples():
        r=f[f.structure_id==chosen.structure_id].iloc[0]; length=r.end-r.start;lo=((r.start-length)//200)*200
        subset=samples[samples.condition.isin(['37C',condition])]
        images=[np.log1p(matrices[s.sample_id][r.structure_id]) for s in subset.itertuples()]
        vmax=float(np.quantile(np.concatenate([x.ravel() for x in images]),.99))
        fig,axes=plt.subplots(2,2,figsize=(8,7))
        for ax,s,a in zip(axes.flat,subset.itertuples(),images):
            ax.imshow(a,origin='lower',cmap='magma',vmin=0,vmax=vmax,extent=[lo/1000,(lo+len(a)*100)/1000]*2)
            for edge in [r.start/1000,r.end/1000]:ax.axvline(edge,color='cyan',lw=.7);ax.axhline(edge,color='cyan',lw=.7)
            ax.set_title(f'{s.condition} rep{int(s.replicate)}');ax.set_xlabel('Genome coordinate (kb)')
        fig.suptitle(f'{r.structure_id}: local log2 change {chosen.log2_enrichment_change:.2f}');fig.tight_layout();fig.savefig(OUT/f'{condition}_{r.structure_id}.png',dpi=140);plt.close(fig)
print(pd.DataFrame(summaries).to_string(index=False),flush=True)

"""Check flank-width, repeat-noise, empirical-null, and heatmap robustness."""
import sys, hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dlpro.experiment_core import ContactReader, save_json
from dlpro.data import overlap_groups

ROOT=Path('results/structure_conditions'); CACHE=Path('data/processed/structure_conditions'); EXPANDED=CACHE/'double_flank'
ROOT.mkdir(exist_ok=True);EXPANDED.mkdir(exist_ok=True)
f=pd.read_csv('data/annotations/structures.csv')
s=pd.read_csv('data/annotations/samples.csv')
s=s[s.condition.isin(['37C','HNS_mut','FIS_mut'])].sort_values(['condition','replicate'])
assert len(f)==344 and len(s)==6
lengths=(f.end-f.start).to_numpy(); ids=f.structure_id.to_numpy()
# Prespecified flank widths: one half, one, and two times target length.
widths=(.5,1.,2.)
# Connect overlapping largest 2x-flank contexts so the empirical noise benchmark respects spatial dependence.
ext=f[['chrom']].copy();ext['start']=((f.start-2*lengths)//200)*200;ext['end']=((f.end+2*lengths+199)//200)*200
f['context_group']=overlap_groups(ext).to_numpy(); n_groups=int(f.context_group.nunique())
# Map every 100bp bin whose center intersects an annotated interval.
all_ann=np.zeros(464166,dtype=bool)
for r in f.itertuples(): all_ann[max(0,int(r.start)//100):min(len(all_ann),int(np.ceil(r.end/100)))]=True

# Expanded 2x-flank matrices; old 1x windows are already cached and cover narrower widths.
expanded={}
for row in s.itertuples():
    path=EXPANDED/(row.sample_id+'.npz')
    if not path.exists():
        reader=ContactReader(row.path); arrays={}
        for i,r in enumerate(f.itertuples()):
            L=int(r.end-r.start);lo=((int(r.start)-2*L)//100)*100;hi=int(np.ceil((int(r.end)+2*L)/100))*100
            if lo<0 or hi>4641652: raise ValueError(f'Expanded context crosses chromosome edge: {r.structure_id}')
            arrays[r.structure_id]=reader.window(lo//100,(hi-lo)//100,100)
            if i%100==0: print(row.sample_id,i,flush=True)
        np.savez_compressed(path,**arrays)
    expanded[row.sample_id]=np.load(path)

# Per-sample metrics keyed by annotation, width, background type.
metrics={}
for r in f.itertuples():
    L=int(r.end-r.start)
    for sample in s.itertuples():
        z=expanded[sample.sample_id][r.structure_id];lo=((int(r.start)-2*L)//100)*100
        centers=lo+(np.arange(len(z))+.5)*100
        core=(centers>=r.start)&(centers<r.end)
        for mult in widths:
            w=mult*L
            left=(centers>=r.start-w)&(centers<r.start)
            right=(centers>=r.end)&(centers<r.end+w)
            flank=(left|right)
            other_ann=all_ann[(centers//100).astype(int)]
            clean_flank=flank&~other_ann
            # The distance range is identical across flank widths (<= shortest flank).
            dmax=max(2,int(np.floor(min(w,.5*L)/100)))
            for clean in (False,True):
                bg=clean_flank if clean else flank
                total=background=weight=0.;nd=0
                for d in range(2,dmax+1):
                    if d>=len(z):break
                    vals=np.diag(z,d);target=core[:-d]&core[d:]
                    bgpairs=bg[:-d]&bg[d:]
                    if target.any() and bgpairs.any():
                        n=int(target.sum()); bgmean=float(vals[bgpairs].mean())
                        if bgmean>0:
                            total+=float(vals[target].sum());background+=bgmean*n;weight+=n;nd+=1
                E=total/background if background>0 and total>0 else np.nan
                metrics[(r.structure_id,sample.sample_id,mult,clean)]=(E,total/weight if weight else np.nan,int(weight),nd,int(clean_flank.sum()))

rows=[]; within=[]
for i,r in f.iterrows():
    for cond in ['HNS_mut','FIS_mut']:
        row=dict(structure_id=r.structure_id,type=r.type,chrom=r.chrom,start=int(r.start),end=int(r.end),context_group=int(r.context_group),condition=cond)
        for mult in widths:
            for clean in (False,True):
                tag=f'w{mult:g}_'+('clean' if clean else 'all')
                wt=np.array([metrics[(r.structure_id,f'GSE272159_37C_rep{rep}',mult,clean)][0] for rep in [1,2]])
                mut_ids=[s[(s.condition==cond)&(s.replicate==rep)].iloc[0].sample_id for rep in [1,2]]
                mutant=np.array([metrics[(r.structure_id,mid,mult,clean)][0] for mid in mut_ids])
                if np.all(np.isfinite(wt)) and np.all(np.isfinite(mutant)) and np.all(wt>0) and np.all(mutant>0):
                    wl=np.log2(wt);ml=np.log2(mutant);cross=(ml[:,None]-wl[None,:]).ravel()
                    effect=float(ml.mean()-wl.mean());all_same=bool(np.all(cross>0) or np.all(cross<0));min_cross=float(np.min(abs(cross)))
                    wt_diff=float(abs(wl[1]-wl[0]));mut_diff=float(abs(ml[1]-ml[0]))
                else: effect=np.nan;all_same=False;min_cross=np.nan;wt_diff=np.nan;mut_diff=np.nan
                row.update({tag+'_effect':effect,tag+'_all_four_same_direction':all_same,tag+'_minimum_cross_effect':min_cross,tag+'_WT_repeat_difference':wt_diff,tag+'_mutant_repeat_difference':mut_diff})
                if clean:
                    for sample in s.itertuples():
                        if sample.condition==cond or sample.condition=='37C':
                            E=metrics[(r.structure_id,sample.sample_id,mult,clean)][0]
                            row[f'{tag}_{sample.condition}_rep{int(sample.replicate)}_enrichment']=E
                if np.isfinite(wt_diff):within.append(dict(structure_id=r.structure_id,context_group=int(r.context_group),condition='37C',flank_width=mult,background='clean' if clean else 'all',abs_repeat_log2=wt_diff))
                if np.isfinite(mut_diff):within.append(dict(structure_id=r.structure_id,context_group=int(r.context_group),condition=cond,flank_width=mult,background='clean' if clean else 'all',abs_repeat_log2=mut_diff))
        rows.append(row)
g=pd.DataFrame(rows);wn=pd.DataFrame(within)
# Empirical background: per context group, take the maximum repeat difference,
# pooling the three within-condition replicate pairs. This handles local overlap
# and is a descriptive noise reference, not a permutation p-value.
null_rows=[]
for mult in widths:
    for bg in ['clean','all']:
        x=wn[(wn.flank_width==mult)&(wn.background==bg)]
        maxima=x.groupby(['condition','context_group']).abs_repeat_log2.max().reset_index()
        for cond in ['HNS_mut','FIS_mut']:
            effects=g[g.condition==cond][f'w{mult:g}_{bg}_effect'].abs()
            valid=effects[np.isfinite(effects)]
            null=maxima.abs_repeat_log2.to_numpy()
            null_rows.append(dict(condition=cond,flank_width=mult,background=bg,empirical_null_groups=len(null),within_repeat_q95=float(np.quantile(null,.95)),within_repeat_q99=float(np.quantile(null,.99)),comparisons=int(len(valid)),above_q95=int((valid>np.quantile(null,.95)).sum()),above_q99=int((valid>np.quantile(null,.99)).sum()),above_max=int((valid>null.max()).sum()),largest_null=float(null.max())))
null=pd.DataFrame(null_rows)
# Main robustness criterion: all widths, both background definitions, all four
# unpaired cross-repeat contrasts and minimum 0.25 log2 effect.
for cond in ['HNS_mut','FIS_mut']:
    ix=g.condition==cond
    core=None
    for mult in widths:
        for bg in ['clean','all']:
            tag=f'w{mult:g}_{bg}_'
            ok=(g[tag+'effect'].abs()>=.25)&g[tag+'all_four_same_direction']&(g[tag+'minimum_cross_effect']>=.25)
            core=ok if core is None else core&ok
    g.loc[ix,'all_widths_backgrounds_robust']=core.loc[ix]
# Join shape-independent primary hits from initial protocol and add empirical null margins.
initial=pd.read_csv(ROOT/'structure_changes.csv')
initial=initial[['structure_id','condition','evidence_qualified','shape_repeat_separation']]
g=g.merge(initial,on=['structure_id','condition'],validate='one_to_one')
for cond in ['HNS_mut','FIS_mut']:
    ix=g.condition==cond
    reference=wn[(wn.flank_width==1.)&(wn.background=='clean')].groupby(['condition','context_group']).abs_repeat_log2.max()
    q95=float(reference.quantile(.95));q99=float(reference.quantile(.99));mx=float(reference.max())
    g.loc[ix,'empirical_noise_q95']=q95;g.loc[ix,'empirical_noise_q99']=q99;g.loc[ix,'empirical_noise_max']=mx
    g.loc[ix,'exceeds_empirical_noise_q99']=g.loc[ix,'w1_clean_effect'].abs()>q99
    g.loc[ix,'exceeds_all_observed_empirical_noise']=g.loc[ix,'w1_clean_effect'].abs()>mx
    all_q99=pd.Series(True,index=g.index);all_max=pd.Series(True,index=g.index)
    for mult in widths:
        for bg in ['clean','all']:
            cutoff=float(null[(null.condition==cond)&(null.flank_width==mult)&(null.background==bg)].within_repeat_q99.iloc[0])
            all_q99 &= g[f'w{mult:g}_{bg}_effect'].abs()>cutoff
            max_cutoff=float(null[(null.condition==cond)&(null.flank_width==mult)&(null.background==bg)].largest_null.iloc[0])
            all_max &= g[f'w{mult:g}_{bg}_effect'].abs()>max_cutoff
    g.loc[ix,'exceeds_noise_q99_at_all_sensitivities']=all_q99.loc[ix]
    g.loc[ix,'exceeds_largest_observed_noise_at_all_sensitivities']=all_max.loc[ix]
# Final short-list requires initial criteria, background sensitivity, and a repeat-noise margin at every tested scale.
g['robust_candidate']=g.evidence_qualified & g.all_widths_backgrounds_robust & g.exceeds_noise_q99_at_all_sensitivities & g.exceeds_largest_observed_noise_at_all_sensitivities

OUT=ROOT/'robustness';OUT.mkdir(exist_ok=True)
base_protocol=pd.read_json(ROOT/'protocol.json',typ='series').to_dict()
save_json(OUT/'protocol.json',dict(extends='results/structure_conditions/protocol.json',conditions=['37C','HNS_mut','FIS_mut'],structures=344,flank_width_multipliers=list(widths),bin_bp=100,common_target_distance_range='200bp through 0.5 times annotated structure length for all flank widths',backgrounds={'all':'all flank pixels','clean':'exclude bins overlapping any other known annotated structure'},repeat_design='two unpaired biological replicates per condition; evaluate all four cross-condition contrasts',overlap_groups=int(n_groups),empirical_null='Within-condition replicate absolute log2 enrichment differences; maximum per largest-context overlap group and condition, pooled across WT, HNS_mut, FIS_mut; 243 group maxima per sensitivity setting',multiple_screening='Threshold comparison against empirical repeat-noise q99 at each flank/background setting; descriptive filter, not formal FDR or family-wise error control',candidate_rule='Previous candidates must preserve direction and abs(log2 change)>=0.25 across all six sensitivities, all four cross-condition comparisons direction-concordant, and every sensitivity exceeds its own pooled empirical-noise q99.',input_samples=base_protocol.get('source_samples',[])))
g.to_csv(OUT/'all_structure_sensitivities.csv',index=False);wn.to_csv(OUT/'within_condition_repeat_noise.csv',index=False);null.to_csv(OUT/'empirical_noise_summary.csv',index=False)
g[g.robust_candidate].sort_values(['condition','w1_clean_effect']).to_csv(OUT/'robust_candidates.csv',index=False)
# Report fixed focal cases even when not passing every check.
focal=[('OPCID_10','HNS_mut'),('OPCID_10','FIS_mut'),('OPCID_15','FIS_mut'),('CHIN_229','HNS_mut')]
g[g.set_index(['structure_id','condition']).index.isin(focal)].to_csv(OUT/'focal_cases.csv',index=False)
summary=[]
for cond in ['HNS_mut','FIS_mut']:
    q=g[g.condition==cond]
    summary.append(dict(condition=cond,all_initial_candidates=int(q.evidence_qualified.sum()),all_widths_both_backgrounds=int(q.all_widths_backgrounds_robust.sum()),exceed_empirical_q99=int(q.exceeds_empirical_noise_q99.sum()),robust_candidates=int(q.robust_candidate.sum()),at_least_one_spatial_group=int(q.loc[q.robust_candidate,'context_group'].nunique())))
save_json(OUT/'summary.json',dict(flank_multipliers=list(widths),backgrounds=['all flanking bins','exclude bins overlapping any other known structure'],groups=n_groups,condition_summary=summary,empirical_null='For each flank/background, take the maximum absolute replicate log2 difference within each adaptive-context overlap group and each of WT, HNS_mut, FIS_mut; pool these group maxima as a descriptive repeat-noise reference. This is not a permutation test or formal p-value.',selection='An initial candidate must maintain direction and at least 0.25 log2 magnitude across all three flank widths and both background definitions; all four unpaired cross-condition contrasts must agree; and each sensitivity effect must exceed both its pooled within-repeat group-maximum 99th-percentile threshold and the largest observed within-repeat group maximum.',limitations=['Only two biological replicates per condition.','Condition-internal replicate differences are one observed realization per structure; spatial groups are dependent despite overlap clustering.','The empirical null is a conservative descriptive noise benchmark, not formal family-wise error control or FDR.','The coordinate data and loci have been examined previously; this is retrospective robustness analysis.']))
# Direct matched heatmaps for requested focal cases; same color scale across the four samples.
for sid,cond in focal:
    rr=f[f.structure_id==sid].iloc[0];sub=s[s.condition.isin(['37C',cond])]
    ims=[]
    for sm in sub.itertuples():
        z=expanded[sm.sample_id][sid];L=int(rr.end-rr.start);lo=((int(rr.start)-2*L)//100)*100
        ims.append(np.log1p(z))
    vmax=float(np.quantile(np.concatenate([x.ravel() for x in ims]),.99))
    fig,axs=plt.subplots(2,2,figsize=(9,7))
    for ax,sm,a in zip(axs.flat,sub.itertuples(),ims):
        lo=((int(rr.start)-2*int(rr.end-rr.start))//100)*100
        ax.imshow(a,origin='lower',cmap='magma',vmin=0,vmax=vmax,extent=[lo/1000,(lo+len(a)*100)/1000]*2)
        ax.axvspan(rr.start/1000,rr.end/1000,color='cyan',alpha=.08);ax.axvline(rr.start/1000,color='cyan',lw=.8);ax.axvline(rr.end/1000,color='cyan',lw=.8);ax.axhline(rr.start/1000,color='cyan',lw=.8);ax.axhline(rr.end/1000,color='cyan',lw=.8)
        ax.set_title(f'{sm.condition} rep{int(sm.replicate)}');ax.set_xlabel('Genome coordinate (kb)')
    eff=g[(g.structure_id==sid)&(g.condition==cond)]['w1_clean_effect'].iloc[0]
    fig.suptitle(f'{sid}, {cond}: clean local-enrichment log2 change {eff:+.2f}')
    fig.tight_layout();fig.savefig(OUT/f'{sid}_{cond}.png',dpi=150);plt.close(fig)
print(pd.DataFrame(summary).to_string(index=False));print(null.to_string(index=False));print('Focal cases:');print(g[g.set_index(['structure_id','condition']).index.isin(focal)][['structure_id','condition','w0.5_clean_effect','w1_clean_effect','w2_clean_effect','robust_candidate','exceeds_empirical_noise_q99']].to_string(index=False),flush=True)

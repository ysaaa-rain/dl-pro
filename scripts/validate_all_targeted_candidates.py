"""Uniform perturbation measurements and WT heatmaps for every selected center."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dlpro.coverage_support import coarse_support
from dlpro.experiment_core import ContactReader,centered_indices,CHROM,LENGTH
from dlpro.method_review import load_config,sha256,write_json
from dlpro.targeted_review import pearson_all

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();cfg=load_config(a.config)
    out=Path(cfg['output'])/'conditions';out.mkdir(exist_ok=True);assert not (out/'sample_measurements.csv').exists()
    directory=Path(cfg['output'])/cfg['discovery']['result_subdir'];selected=pd.read_csv(directory/'selected_scan_windows.csv')
    quality=pd.read_csv(directory/'candidate_quality_per_window.csv');quality.to_csv(out/'all_ranked_candidate_evidence.csv',index=False)
    membership=pd.read_csv(directory/'cluster_membership.csv')
    catalog=quality.merge(membership,on=['hidden_type','repeat','method','global_equivalent_budget','center'],validate='one_to_many')
    catalog['chrom']=CHROM
    catalog['start']=[centered_indices(int(c))[1]*cfg['discovery']['input_scales_bp'][-1] for c in catalog.center]
    catalog['length_bp']=cfg['discovery']['max_context_bp'];catalog['end']=catalog.start+catalog.length_bp
    catalog['interval_semantics']='Maximum candidate input context, not inferred biological structure extent'
    catalog['distinction_from_known']=np.where(catalog.full_known_context_overlap,'Known annotation overlaps complete input context','No annotated context overlap; not proof of biological novelty')
    annotations=pd.read_csv('data/annotations/structures.csv')
    gaps=[]
    for row in catalog.itertuples():
        gap=min(float(np.maximum(np.maximum(annotations.start.to_numpy()+shift-row.end,row.start-annotations.end.to_numpy()-shift),0).min()) for shift in (-LENGTH,0,LENGTH))
        gaps.append(gap)
    catalog['nearest_annotation_context_gap_bp']=gaps
    delta=abs(catalog.center.to_numpy()[:,None]-annotations.center.to_numpy()[None])
    catalog['nearest_known_center_distance_bp']=np.minimum(delta,LENGTH-delta).min(1)
    catalog['exclusion_reason']=np.where(catalog.full_known_context_overlap,'known_context',np.where(~catalog.passes_repeat_qc,'repeat_quality',np.where(~catalog.independent_unannotated,'nonindependent','none_at_window_level')))
    catalog.to_csv(out/'candidate_catalog.csv',index=False)
    # Select the reference recipe by a deterministic key, never by favorable condition results.
    centers=selected.sort_values(['hidden_type','fold','center']).drop_duplicates('center').sort_values('center')
    centers.to_csv(out/'unique_center_reference_recipes.csv',index=False)
    samples=pd.read_csv('data/annotations/samples.csv');wanted=samples[samples.condition.isin(['37C']+cfg['diagnostic']['condition_names'])].sort_values(['condition','replicate'])
    cache=Path('data/processed')/cfg['version'];cache.mkdir(exist_ok=True)
    matrices={};sources=[];unavailable=[];n=cfg['diagnostic']['condition_bins'];bp=cfg['diagnostic']['condition_bp']
    for sample in wanted.itertuples():
        if not Path(sample.path).exists():
            unavailable.append(dict(sample_id=sample.sample_id,reason='Complete raw Cooler unavailable; existing selected historical candidate cache cannot cover new full list'));continue
        assert sha256(sample.path)==sample.sha256
        path=cache/f'candidate_{sample.sample_id}.npy';assert not path.exists()
        array=np.lib.format.open_memmap(path,mode='w+',dtype='float32',shape=(len(centers),n,n));reader=ContactReader(sample.path)
        for pos,row in enumerate(centers.itertuples()):array[pos]=reader.window(centered_indices(int(row.center))[0],n,bp)
        array.flush();matrices[sample.sample_id]=array
        sources.append(dict(sample_id=sample.sample_id,condition=sample.condition,replicate=int(sample.replicate),source_sha256=sample.sha256,cache=path.as_posix(),cache_sha256=sha256(path)))
        print('All candidates condition',sample.sample_id,len(centers),flush=True)
    wt=wanted[wanted.condition=='37C'].sort_values('replicate');assert len(wt)==2
    support={r:coarse_support(np.load(f'data/processed/coverage_rep{r}.npz')['coverage'],centers.center.to_numpy())[:,0]>=cfg['discovery']['support_fraction'] for r in (1,2)}
    distance=abs(np.arange(n)[:,None]-np.arange(n)[None]);triangle=np.triu(np.ones((n,n),bool),2)
    measurements=[];comparisons=[];(out/'heatmaps').mkdir(exist_ok=True)
    source_table={x['sample_id']:x for x in sources}
    for pos,row in enumerate(centers.itertuples()):
        expected=np.load(directory/f'{row.hidden_type}_fold{row.fold}/transform.npz')['expected'][0,distance]
        wt_arrays=[matrices[sample.sample_id][pos] for sample in wt.itertuples()]
        wt_mask=triangle&support[1][pos,:,None]&support[1][pos,None,:]&support[2][pos,:,None]&support[2][pos,None,:]
        wt_mean=np.mean(wt_arrays,axis=0);wt_fields=[];sample_fields={};signals={}
        for sample in wanted.itertuples():
            if sample.sample_id not in matrices:continue
            image=matrices[sample.sample_id][pos];valid_rows=image.sum(-1)>0;mask=wt_mask&valid_rows[:,None]&valid_rows[None,:]
            oe=image/expected;wt_oe=wt_mean/expected
            amplitude=max(float(oe[mask].mean()),1e-12) if mask.any() else 1
            wt_amplitude=max(float(wt_oe[mask].mean()),1e-12) if mask.any() else 1
            field=np.log1p(oe/amplitude);wt_field=np.log1p(wt_oe/wt_amplitude)
            signal=float(image[mask].mean()) if mask.any() else None;wt_signal=float(wt_mean[mask].mean()) if mask.any() else None
            record=dict(center=int(row.center),reference_hidden_type=row.hidden_type,reference_fold=int(row.fold),sample_id=sample.sample_id,condition=sample.condition,replicate=int(sample.replicate),effective_triangle_fraction=float(mask.sum()/triangle.sum()),mean_density_per_million=signal,signal_ratio_vs_matched_wt=signal/max(wt_signal,1e-12) if mask.any() else None,log_shape_correlation_vs_wt=pearson_all(field[mask],wt_field[mask]),log_shape_rmse_vs_wt=float(np.sqrt(((field-wt_field)[mask]**2).mean())) if mask.any() else None)
            measurements.append(record);sample_fields[sample.sample_id]=field;signals[sample.sample_id]=signal
            if sample.condition=='37C':wt_fields.append((sample.sample_id,field))
        for condition in sorted(wanted.condition.unique()):
            ids=[x.sample_id for x in wanted[wanted.condition==condition].itertuples() if x.sample_id in matrices]
            comparisons.append(dict(center=int(row.center),condition=condition,available_repeats=len(ids),repeat_shape_pearson=pearson_all(sample_fields[ids[0]][wt_mask],sample_fields[ids[1]][wt_mask]) if len(ids)==2 else None,descriptive_only=True))
        fig,axes=plt.subplots(1,2,figsize=(7,3.4),constrained_layout=True)
        for ax,(name,field) in zip(axes,wt_fields):
            image=ax.imshow(field,origin='lower',cmap='magma',vmin=0,vmax=4);ax.set_title(name.replace('GSE272159_',''),fontsize=9);ax.set_xlabel('100 bp bin')
        fig.suptitle(f'All selected candidates: center {int(row.center):,}')
        fig.colorbar(image,ax=axes,label='Amplitude-normalized log(1+O/E)',shrink=.85)
        fig.savefig(out/'heatmaps'/f'center_{int(row.center)}.png',dpi=100);plt.close(fig)
    pd.DataFrame(measurements).to_csv(out/'sample_measurements.csv',index=False);pd.DataFrame(comparisons).to_csv(out/'repeat_condition_comparisons.csv',index=False)
    pd.DataFrame(measurements).groupby('condition')[['signal_ratio_vs_matched_wt','log_shape_correlation_vs_wt','log_shape_rmse_vs_wt','effective_triangle_fraction']].agg(['mean','median','count']).to_csv(out/'condition_summary.csv')
    write_json(out/'protocol.json',dict(unique_selected_centers=len(centers),ranked_records=len(selected),all_selected_centers_measured=True,wt_heatmaps=len(list((out/'heatmaps').glob('*.png'))),sources=sources,unavailable_sources=unavailable,common_display_range=[0,4],reference='Lexicographically first hidden-type recipe for each OOF center; same reference for every condition at this center',intensity='Depth-normalized counts per million; reported separately from amplitude-normalized log shape',mask='WT common native80percent support; condition nonzero coarse rows; effective fraction reported per sample',NoLig='Single sample, descriptive only; no significance or causal claim',selection='Frozen ranked list; no condition data used in scoring or filtering'))

if __name__=='__main__':main()

"""Genome-wide four-panel tracks, three conditions, library-normalized local contacts."""
import argparse,concurrent.futures,json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from dlpro.experiment_core import ContactReader,LENGTH,save_json

OUT=Path('results/multitrack')
COLORS={'37C':'#22577A','44C':'#D66B38','rif25':'#3D8361','HNS_mut':'#D66B38','FIS_mut':'#3D8361'}
STRUCT_COLORS={'CHIN':'#F5A623','OPCID':'#B96522','CHID':'#E07A35'}


def draw_region(args):
    start,end,centers,tracks,genes,structures,output=args
    active=structures[(structures.start<end)&(structures.end>start)].sort_values('start')
    lanes=[];packed=[]
    for r in active.itertuples():
        lane=next((i for i,x in enumerate(lanes) if x<=r.start),len(lanes))
        if lane==len(lanes):lanes.append(r.end)
        else:lanes[lane]=r.end
        packed.append((r,lane))
    bottom_height=max(1,len(lanes)*.28)
    fig,axes=plt.subplots(4,1,sharex=True,figsize=(12,4.3+bottom_height),gridspec_kw={'height_ratios':[2,1,1,bottom_height]},constrained_layout=True)
    mask=(centers>=start)&(centers<end);xx=centers[mask]
    mean=np.mean(np.stack(list(tracks.values())),axis=0)[mask]
    axes[0].fill_between(xx,0,np.sqrt(mean),color='gray',alpha=.15,label='Condition mean')
    for name,signal in tracks.items():axes[0].plot(xx,np.sqrt(signal[mask]),lw=1,color=COLORS[name],label=name)
    axes[0].legend(ncol=4,fontsize=7,loc='upper right');axes[0].set_ylabel('sqrt(local contact\ndensity / million pairs)',fontsize=8)
    axes[0].set_title(f'NC_000913.3: {start:,}-{end:,} bp; 100 bp bins, +/-10 kb contacts',fontsize=10)
    axes[1].fill_between(xx,0,mean,color='#86939F',alpha=.6);axes[1].set_ylabel('Mean signal',fontsize=8)
    sub=genes[(genes.start<end)&(genes.end>start)]
    for r in sub.itertuples():
        yy=.7 if r.strand=='+' else .2
        lo=max(start,r.start);hi=min(end,r.end)
        axes[2].add_patch(Rectangle((lo,yy),hi-lo,.17,color='#3479B5',alpha=.8))
        if hi-lo>300:axes[2].text((lo+hi)/2,yy+.18,r.name,ha='center',fontsize=7,clip_on=True)
    axes[2].set(ylim=(0,1.3),yticks=[.3,.8],yticklabels=['-','+']);axes[2].set_ylabel('Genes',fontsize=8)
    for r,lane in packed:
        lo=max(start,r.start);hi=min(end,r.end)
        axes[3].add_patch(Rectangle((lo,lane),hi-lo,.6,color=STRUCT_COLORS[r.type],alpha=.75))
        if hi-lo>=500:axes[3].text((lo+hi)/2,lane+.63,r.structure_id,ha='center',fontsize=7,clip_on=True)
    axes[3].set(ylim=(-.1,max(1,len(lanes))+.6),yticks=[],xlim=(start,end));axes[3].set_ylabel('Structures',fontsize=8);axes[3].set_xlabel('Genomic position (bp)')
    for ax in axes:ax.tick_params(labelsize=7);ax.spines[['top','right']].set_visible(False)
    path=Path(output)/'tracks'/f'{start:07d}_{end:07d}.png';fig.savefig(path,dpi=100);plt.close(fig)
    return dict(start=start,end=end,path=str(path),genes=len(sub),structures=len(active))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--conditions',nargs='+',default=['37C','44C','rif25']);parser.add_argument('--output',default='results/multitrack');args=parser.parse_args()
    global OUT
    OUT=Path(args.output)
    if len(args.conditions)<3 or len(set(args.conditions))!=len(args.conditions) or not set(args.conditions)<=set(COLORS):raise ValueError('Choose at least three distinct supported conditions')
    OUT.mkdir(exist_ok=True,parents=True);(OUT/'tracks').mkdir(exist_ok=True)
    samples=pd.read_csv('data/annotations/samples.csv');n=(LENGTH+99)//100
    selected=samples[samples.condition.isin(args.conditions)]
    if not selected.groupby('condition').size().reindex(args.conditions).eq(2).all():raise ValueError('Exactly two repeats per condition required')
    source_signature={r.sample_id:r.sha256 for r in selected.itertuples()}
    manifest=OUT/'cache_sources.json'
    if manifest.exists() and json.loads(manifest.read_text())!=source_signature:raise ValueError('Source mismatch: use a new output directory')
    cache=OUT/'signals.npz';records=[]
    if cache.exists():
        obj=np.load(cache);signals={k:obj[k] for k in obj.files}
        if set(signals)!=set(source_signature):raise ValueError('Cached samples differ: use a new output directory')
    else:
        signals={}
        for sample in selected.itertuples():
            reader=ContactReader(sample.path);signal=np.zeros(n,np.float32)
            for start in range(0,n,400):
                count=min(400,n-start);a=reader.window(start-100,n=count+200,bin_bp=100)
                for j in range(count):
                    row=j+100;signal[start+j]=a[row,row-100:row+101].sum()-a[row,row]
                if start%8000==0:print(f'tracks {sample.sample_id}: {start}/{n}',flush=True)
            signals[sample.sample_id]=signal
            records.append(dict(sample_id=sample.sample_id,condition=sample.condition,replicate=int(sample.replicate),library_pairs=reader.library_size,path=sample.path))
        np.savez_compressed(cache,**signals)
        pd.DataFrame(records).to_csv(OUT/'sample_depths.csv',index=False)
    manifest.write_text(json.dumps(source_signature,indent=2)+'\n')
    tracks={condition:np.mean([signals[r.sample_id] for r in selected[selected.condition==condition].itertuples()],axis=0) for condition in args.conditions}
    centers=np.arange(n)*100+np.minimum(100,LENGTH-np.arange(n)*100)/2
    table=pd.DataFrame({'start':np.arange(n)*100,'end':np.minimum(np.arange(n)*100+100,LENGTH),**tracks})
    table.to_csv(OUT/'signals.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    genes=pd.read_csv('data/annotations/genes.csv');structures=pd.read_csv('data/annotations/structures.csv')
    jobs=[(s,min(s+10000,LENGTH),centers,tracks,genes,structures,str(OUT)) for s in range(0,LENGTH,10000)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
        index=[]
        for i,row in enumerate(pool.map(draw_region,jobs)):
            index.append(row)
            if i%100==0:print(f'rendered tracks {i+1}/{len(jobs)}',flush=True)
    frame=pd.DataFrame(index);frame.to_csv(OUT/'index.csv',index=False)
    assert frame.start.iloc[0]==0 and frame.end.iloc[-1]==LENGTH
    assert np.array_equal(frame.start.to_numpy()[1:],frame.end.to_numpy()[:-1])
    save_json(OUT/'protocol.json',dict(conditions=args.conditions,repeats_per_condition=2,bin_bp=100,contact_radius_bp=10000,diagonal_excluded=True,
        local_contact_definition='Sum of symmetric-image contact densities within +/-100 coarse bins, excluding self; per-million library pair normalization',
        repeat_aggregation='Arithmetic mean after depth normalization',panels=['condition curves with gray condition mean','mean signal','reference genes','known structures'],
        interval_bp=10000,n_figures=len(frame),covered_bp=int((frame.end-frame.start).sum()),full_genome_covered=True,circular_origin_supported=True,
        limitations=['local +/-10kb contacts, not total chromosome-wide row coverage','coarse image bins normalized by their actual area; no ICE balancing','same-study condition comparison; signals do not establish mechanism']))

if __name__=='__main__':main()

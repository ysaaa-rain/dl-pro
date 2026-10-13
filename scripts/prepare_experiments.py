"""Prepare known windows and the genome-wide circular scan, from real WT data."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import pandas as pd
from dlpro.experiment_core import ContactReader,plan_known,centered_indices,LENGTH,hits,save_json

parser=argparse.ArgumentParser();parser.add_argument('--reuse-cache',action='store_true');args=parser.parse_args()
Path('data/processed').mkdir(exist_ok=True)
f=plan_known();samples=pd.read_csv('data/annotations/samples.csv')
wt=samples[samples.condition=='37C'].sort_values('replicate')
scan=pd.DataFrame(dict(center=np.minimum(np.arange(0,(LENGTH+99)//100,64)*100,LENGTH-1)))
scan['known_overlap']=[bool(hits(int(c),12800,f,buffer=200).any()) for c in scan.center]
scan['background_block']=scan.center//100000
scan['training_background']=(~scan.known_overlap)&(scan.center%100000>=12800)&(scan.center%100000<87200)
scan.to_csv('data/processed/scan_windows.csv',index=False)
for _,sample in wt.iterrows():
    repeat=int(sample.replicate);reader=None
    for kind,frame in [('known',f),('scan',scan)]:
        dest=Path(f'data/processed/{kind}_rep{repeat}.npz')
        if dest.exists():
            if not args.reuse_cache:raise FileExistsError(f'Use --reuse-cache to verify and reuse {dest}')
            expected=json.loads(Path('data/processed/cache_manifest.json').read_text())
            for source,key in [('data/annotations/structures.csv','annotation_sha256'),('data/processed/known_windows.csv','window_manifest_sha256'),('data/processed/scan_windows.csv','scan_manifest_sha256')]:
                if hashlib.sha256(Path(source).read_bytes()).hexdigest()!=expected[key]:raise ValueError('Cached source or split differs')
            if hashlib.sha256(dest.read_bytes()).hexdigest()!=expected['caches'][str(dest)]['sha256']:raise ValueError('Cached patch checksum differs')
            shape=np.load(dest)['raw'].shape
            if shape!=(len(frame),2,128,128):raise ValueError('Cached shape differs')
            print(f'verified reusable cache {dest}',flush=True);continue
        if reader is None:reader=ContactReader(sample.path)
        matrices=[]
        for i,row in frame.iterrows():
            starts=centered_indices(int(row.center))
            matrices.append(np.stack([reader.window(starts[0],bin_bp=100),reader.window(starts[1],bin_bp=200)]))
            if i%100==0:print(f'{kind} rep{repeat} {i+1}/{len(frame)}',flush=True)
        np.savez_compressed(dest,raw=np.stack(matrices))
        print(f'saved {dest} ({dest.stat().st_size} bytes)',flush=True)
save_json('results/experiment/preprocessing_protocol.json',dict(
    native_bp=10,output_shape=[2,128,128],scales_bp=[100,200],widths_bp=[12800,25600],
    matrix_view='symmetric contact image; area-normalized coarse pixel mean',depth_units='per million library pairs',
    circular_origin_supported=True,last_bin_area_corrected=True,n_scan_windows=len(scan),scan_stride_bp=6400,
    coverage='centers tiled across entire circular chromosome, including last partial interval',
    normalization='No ICE; distance expectation and clipping scale fit only on classification training patches',
    source_records='data/annotations/samples.csv',annotation_resolution='paper BED pinned by commit; course source preserved'))

manifest=dict(annotation_sha256=hashlib.sha256(Path('data/annotations/structures.csv').read_bytes()).hexdigest(),window_manifest_sha256=hashlib.sha256(Path('data/processed/known_windows.csv').read_bytes()).hexdigest(),scan_manifest_sha256=hashlib.sha256(Path('data/processed/scan_windows.csv').read_bytes()).hexdigest(),caches={})
for p in sorted(Path('data/processed').glob('*_rep*.npz')):
    if p.name.startswith(('known_','scan_')):manifest['caches'][str(p)]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
save_json('data/processed/cache_manifest.json',manifest)

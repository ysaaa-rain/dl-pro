"""50bp scanner view for the fixed multiscale classifier, with circular sparse reads."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from dlpro.experiment_core import ContactReader,save_json
scan=pd.read_csv('data/processed/scan_windows.csv');samples=pd.read_csv('data/annotations/samples.csv');wt=samples[samples.condition=='37C'].sort_values('replicate');out=Path('data/processed');signature={'plan_sha256':hashlib.sha256(Path('data/processed/scan_windows.csv').read_bytes()).hexdigest(),'sources':{s.sample_id:s.sha256 for s in wt.itertuples()},'bin_bp':50,'width_bp':6400};manifest=out/'fine_scan_sources.json'
if manifest.exists() and json.loads(manifest.read_text())!=signature:raise ValueError('Fine scan signature differs')
for sample in wt.itertuples():
 path=out/f'fine_scan_rep{int(sample.replicate)}.npz'
 if path.exists():assert np.load(path)['raw'].shape==(len(scan),1,128,128);continue
 reader=ContactReader(sample.path);values=[]
 for i,c in enumerate(scan.center):
  start=int(np.floor((c+25)/50))-64;values.append(reader.window(start,128,50)[None])
  if i%200==0:print('fine scan',int(sample.replicate),i,flush=True)
 np.savez_compressed(path,raw=np.asarray(values,dtype='float32'))
save_json(manifest,signature)

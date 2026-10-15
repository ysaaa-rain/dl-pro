"""Add a50bp central view without expanding the existing25.6kb split extents."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from dlpro.experiment_core import ContactReader,save_json
f=pd.read_csv('data/processed/known_windows.csv');samples=pd.read_csv('data/annotations/samples.csv');wt=samples[samples.condition=='37C'].sort_values('replicate');out=Path('data/processed');signature={'known_plan_sha256':hashlib.sha256(Path('data/processed/known_windows.csv').read_bytes()).hexdigest(),'sources':{r.sample_id:r.sha256 for r in wt.itertuples()},'binsize':50,'width':6400};manifest=out/'fine_windows_sources.json'
if manifest.exists() and json.loads(manifest.read_text())!=signature:raise ValueError('Fine-window source signature changed')
for sample in wt.itertuples():
 path=out/f'fine_rep{int(sample.replicate)}.npz'
 if path.exists():
  obj=np.load(path);assert obj['raw'].shape==(len(f),1,128,128);print('reuse fine',sample.replicate,flush=True);continue
 reader=ContactReader(sample.path);values=[]
 for i,row in enumerate(f.itertuples()):
  start=int(np.floor((row.center+25)/50))-64;assert start*50>=row.window_start and (start+128)*50<=row.window_end
  values.append(reader.window(start,128,50)[None])
  if i%100==0:print('fine extraction',sample.replicate,i,flush=True)
 np.savez_compressed(path,raw=np.asarray(values,dtype='float32'))
save_json(manifest,signature)

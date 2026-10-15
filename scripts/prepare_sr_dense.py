"""Expand only original SR training blocks; preserve validation/test matrices."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from dlpro.experiment_core import ContactReader,save_json
out=Path('data/processed/sr_dense');out.mkdir(exist_ok=True);old=pd.read_csv('data/processed/sr_expanded/windows.csv');train=old[old.split=='train'].copy();extra=[]
for group in sorted(train.group.unique()):
 for offset in range(0,90000,10000):extra.append(dict(group=group,start=group*100000+offset,end=group*100000+offset+10240,split='train',replicate=1,depth_multiplier=1.))
frame=pd.concat([train,pd.DataFrame(extra)],ignore_index=True).drop_duplicates(['start','replicate']).sort_values(['group','start']).reset_index(drop=True);frame['original_sparse']=frame.start.isin(train.start);assert len(frame)==352 and frame.original_sparse.sum()==64
assert ((frame.start//100000)==((frame.end-1)//100000)).all();assert not set(frame.group)&set(old[old.split!='train'].group)
sample=pd.read_csv('data/annotations/samples.csv').query('condition=="37C" and replicate==1').iloc[0];signature=dict(source_sha256=sample.sha256,original_plan_sha256=hashlib.sha256(Path('data/processed/sr_expanded/windows.csv').read_bytes()).hexdigest(),train_windows=352,window_bp=10240,holdout='Original val/test unchanged, all new windows wholly within training100kbblocks')
manifest=out/'sources.json'
if manifest.exists():assert json.loads(manifest.read_text())==signature
frame.to_csv(out/'windows.csv',index=False);reader=ContactReader(sample.path)
rawpath=out/'raw_counts.npy'
if not rawpath.exists():
 raw=np.lib.format.open_memmap(rawpath,mode='w+',dtype='uint32',shape=(len(frame),1024,1024))
 for i,row in frame.iterrows():
  a=reader.window(int(row.start)//10,1024,10,False);assert np.array_equal(a,np.rint(a));raw[i]=a.astype('uint32')
  if i%32==0:print('dense extraction',i,len(frame),flush=True)
 raw.flush();del raw
raw=np.load(rawpath,mmap_mode='r');assert raw.shape==(352,1024,1024)
for start in range(0,len(frame),8):
 path=out/f'counts_{start:03d}.npz'
 if not path.exists():np.savez_compressed(path,raw=raw[start:start+8],start=start)
save_json(manifest,signature);print('Dense train352; original sparse64; validation/test untouched',flush=True)

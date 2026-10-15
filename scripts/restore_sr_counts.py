"""Restore large local count arrays from committed compressed shards."""
from pathlib import Path
import numpy as np
import pandas as pd
for directory in ['sr_expanded','sr_dense']:
 root=Path('data/processed')/directory;frame=pd.read_csv(root/'windows.csv');path=root/'raw_counts.npy'
 if path.exists():
  assert np.load(path,mmap_mode='r').shape==(len(frame),1024,1024);continue
 raw=np.lib.format.open_memmap(path,mode='w+',dtype='uint32',shape=(len(frame),1024,1024));covered=np.zeros(len(frame),bool)
 for shard in sorted(root.glob('counts_*.npz')):
  with np.load(shard) as obj:
   start=int(obj['start']);a=obj['raw'];assert a.shape[1:]==(1024,1024);assert not covered[start:start+len(a)].any();raw[start:start+len(a)]=a;covered[start:start+len(a)]=True
 assert covered.all();raw.flush();print('Restored',path,len(frame),flush=True)

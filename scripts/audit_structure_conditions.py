"""Validate structure-condition artifacts and record input/cache integrity."""
import sys,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
from dlpro.data import overlap_groups
from dlpro.experiment_core import save_json
out=Path('results/structure_conditions');f=pd.read_csv('data/annotations/structures.csv');g=pd.read_csv(out/'structure_changes.csv')
s=pd.read_csv('data/annotations/samples.csv');s=s[s.condition.isin(['37C','HNS_mut','FIS_mut'])]
def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
checks=[]
for row in s.itertuples():
    path=Path('data/processed/structure_conditions')/(row.sample_id+'.npz');z=np.load(path)
    assert set(z.files)==set(f.structure_id)
    assert all(np.isfinite(z[k]).all() and (z[k]>=0).all() and np.allclose(z[k],z[k].T) and len(z[k])%2==0 for k in z.files)
    source=digest(row.path);assert source==row.sha256
    checks.append(dict(sample_id=row.sample_id,source_sha256=source,cache_sha256=digest(path),windows=len(z.files),cache_bytes=path.stat().st_size))
assert len(g)==688 and not g.duplicated(['structure_id','condition']).any()
assert np.isfinite(g.select_dtypes('number')).all().all()
length=f.end-f.start;ext=f[['chrom']].copy();ext['start']=((f.start-length)//200)*200;ext['end']=((f.end+length+199)//200)*200
f['context_group']=overlap_groups(ext)
g=g.drop(columns=['context_group'],errors='ignore').merge(f[['structure_id','context_group']],on='structure_id',validate='many_to_one')
g.to_csv(out/'structure_changes.csv',index=False)
selected=g[g.supported_local_change].copy();selected.to_csv(out/'repeat_consistent_changes.csv',index=False)
save_json(out/'integrity.json',dict(input_and_cache_checks=checks,contrasts=len(g),structures=len(f),adaptive_context_groups=int(f.context_group.nunique()),selected_unique_structures=int(selected.structure_id.nunique()),selected_context_groups=int(selected.context_group.nunique()),all_numeric_fields_finite=True,all_windows_symmetric_nonnegative=True))
usage=pd.read_csv('data/annotations/sample_usage.csv')
for i in usage.index[usage.sample_id.isin(s.sample_id)]:
    tasks=set(str(usage.at[i,'formal_tasks']).split(';'))
    # Existing inventory uses semicolons for multi-task cells.
    tasks.add('4');usage.at[i,'formal_tasks']=';'.join(sorted(tasks))
usage.to_csv('data/annotations/sample_usage.csv',index=False)
print((out/'integrity.json').read_text())

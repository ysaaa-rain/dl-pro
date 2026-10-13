"""Audit zero-contact bins and classifier sensitivity to their image patterns."""
from pathlib import Path
import numpy as np
import pandas as pd
import cooler
import torch
from torch.nn import functional as F
from dlpro.experiment_core import Classifier,ShapeTransform,centered_indices,save_json,LENGTH

out=Path('results/coverage_qc');out.mkdir(parents=True,exist_ok=True)
samples=pd.read_csv('data/annotations/samples.csv');wt=samples[samples.condition=='37C'].sort_values('replicate');coverage={};records=[]
for s in wt.itertuples():
    rep=int(s.replicate);path=Path(f'data/processed/coverage_rep{rep}.npz');c=cooler.Cooler(s.path)
    if path.exists():cov=np.load(path)['coverage']
    else:
        cov=np.zeros(int(c.info['nbins']),dtype='float64')
        for start in range(0,int(c.info['nnz']),2000000):
            p=c.pixels()[start:start+2000000];i=p.bin1_id.to_numpy();j=p.bin2_id.to_numpy();w=p['count'].to_numpy()
            cov+=np.bincount(i,weights=w,minlength=len(cov));cov+=np.bincount(j,weights=w,minlength=len(cov))
            diagonal=i==j;cov-=np.bincount(i[diagonal],weights=w[diagonal],minlength=len(cov))
            if start%50000000==0:print(f'coverage rep{rep}: {start}/{c.info["nnz"]}',flush=True)
        np.savez_compressed(path,coverage=cov)
    if not int(c.info['sum'])<=cov.sum()<=2*int(c.info['sum']):raise ValueError('Coverage row mass disagrees with library count')
    if not np.isfinite(cov).all() or (cov<0).any():raise ValueError('Invalid coverage')
    coverage[rep]=cov;records.append(dict(replicate=rep,n_bins=len(cov),zero_contact_bins=int((cov==0).sum()),fraction_zero=float((cov==0).mean())))
f=pd.read_csv('data/processed/known_windows.csv');tf=ShapeTransform.load('results/classification/transform.npz');model=Classifier();model.load_state_dict(torch.load('results/classification/model_seed42.pt',weights_only=True)['state_dict']);model.eval();torch.set_num_threads(4)
raw_known={r:np.load(f'data/processed/known_rep{r}.npz')['raw'] for r in [1,2]}
rows=[]
for idx,row in f[f.split=='test'].iterrows():
    original=[];corrected=[];missing=[]
    for rep in [1,2]:
        raw=raw_known[rep][idx:idx+1];x=tf.transform(raw);adjusted=x.copy();starts=centered_indices(int(row.center))
        for channel,factor in enumerate([10,20]):
            offset=starts[channel]*factor;coarse_valid=(coverage[rep][offset:offset+128*factor].reshape(128,factor)>0).any(1)
            invalid=(~coarse_valid[:,None])|(~coarse_valid[None,:]);missing.append(float(invalid.mean()))
            adjusted[0,channel,invalid]=np.log(2)/tf.scale[channel]
            diagonal=abs(np.arange(128)[:,None]-np.arange(128)[None,:])<2;adjusted[0,channel,diagonal]=0
        with torch.no_grad():original.append(F.softmax(model(torch.from_numpy(x)),1).numpy()[0]);corrected.append(F.softmax(model(torch.from_numpy(adjusted)),1).numpy()[0])
    p0=np.mean(original,axis=0);p1=np.mean(corrected,axis=0)
    rows.append(dict(structure_id=row.structure_id,type=row.type,mean_zero_coverage_pixel_fraction=float(np.mean(missing)),prediction_changed=bool(p0.argmax()!=p1.argmax()),probability_l1_change=float(abs(p1-p0).sum())))
pd.DataFrame(rows).to_csv(out/'neutral_fill_sensitivity.csv',index=False)
save_json(out/'summary.json',dict(coverage=records,n_test_loci=len(rows),prediction_changes=sum(x['prediction_changed'] for x in rows),mean_probability_l1_change=float(np.mean([x['probability_l1_change'] for x in rows])),
    interpretation='Descriptive neutral-fill sensitivity only; zero counts do not establish mapping failure or biological absence',model='Seed42 original classifier; no retraining or model selection'))

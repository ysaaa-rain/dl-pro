"""Amplitude-invariant shape and observed-support checks for condition contrasts."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
from dlpro.experiment_core import save_json
OUT=Path('results/structure_conditions')
f=pd.read_csv('data/annotations/structures.csv')
s=pd.read_csv('data/annotations/samples.csv');s=s[s.condition.isin(['37C','HNS_mut','FIS_mut'])]
z={r.sample_id:np.load(Path('data/processed/structure_conditions')/(r.sample_id+'.npz')) for r in s.itertuples()}
rows=[]
for r in f.itertuples():
    length=r.end-r.start;lo=((r.start-length)//200)*200
    vectors={};support={}
    for sample in s.itertuples():
        a=z[sample.sample_id][r.structure_id].astype(float);centers=lo+(np.arange(len(a))+.5)*100
        core=(centers>=r.start)&(centers<r.end);left=centers<r.start;right=centers>=r.end
        support[(sample.condition,int(sample.replicate))]=float((a.sum(axis=1)[core]>0).mean())
        values=[]
        for d in range(2,min(len(a),length//100+1)):
            diag=np.diag(a,d);target=core[:-d]&core[d:];flank=(left[:-d]&left[d:])|(right[:-d]&right[d:])
            if target.any() and flank.any():
                # Positive small floor is only a numerical safeguard for sparse flank means.
                values.extend(diag[target]/max(float(diag[flank].mean()),1e-10))
        vector=np.asarray(values);vector=np.log1p(vector/max(vector.mean(),1e-10))
        vectors[(sample.condition,int(sample.replicate))]=vector
    for condition in ['HNS_mut','FIS_mut']:
        wt=[vectors[('37C',i)] for i in [1,2]];mut=[vectors[(condition,i)] for i in [1,2]]
        rmse=lambda a,b:float(np.sqrt(np.mean((a-b)**2)))
        corr=lambda a,b:float(np.corrcoef(a,b)[0,1]) if min(np.std(a),np.std(b))>0 else 0.
        within=max(rmse(*wt),rmse(*mut));cross=[rmse(a,b) for a in wt for b in mut]
        rows.append(dict(structure_id=r.structure_id,condition=condition,wt_shape_repeat_correlation=corr(*wt),mutant_shape_repeat_correlation=corr(*mut),maximum_within_shape_rmse=within,minimum_cross_shape_rmse=min(cross),mean_cross_shape_rmse=float(np.mean(cross)),shape_repeat_separation=bool(min(cross)>1.25*within and min(corr(*wt),corr(*mut))>=.5),minimum_core_observed_row_fraction=min(support[(c,i)] for c in ['37C',condition] for i in [1,2])))
a=pd.DataFrame(rows);a.to_csv(OUT/'shape_and_support.csv',index=False)
g=pd.read_csv(OUT/'structure_changes.csv').merge(a,on=['structure_id','condition'],validate='one_to_one')
g['supported_local_change']=g.evidence_qualified & (g.minimum_core_observed_row_fraction>=.8)
g.to_csv(OUT/'structure_changes.csv',index=False)
g[g.supported_local_change].sort_values('log2_enrichment_change').to_csv(OUT/'repeat_consistent_changes.csv',index=False)
save_json(OUT/'shape_summary.json',dict(conditions=[dict(condition=c,shape_separated=int(t.shape_repeat_separation.sum()),supported_local_change=int(t.supported_local_change.sum()),both=int((t.shape_repeat_separation&t.supported_local_change).sum()),by_type=t.groupby('type').supported_local_change.sum().astype(int).to_dict()) for c,t in g.groupby('condition')],definition='Core log1p O/E vectors normalized by their mean amplitude; each sample expected is its own distance-matched flank mean. All four cross-condition RMSE must exceed 1.25 times the larger within-condition RMSE; both repeat correlations >=0.5.',limitations='Descriptive shape separation, not a hypothesis test. Observed row support is not independently measured mappability. Flank means are local and can be noisy or contain structures.'))
print((OUT/'shape_summary.json').read_text())

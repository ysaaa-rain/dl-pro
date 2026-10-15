"""Measure centered-input collisions and nested-label proximity without changing labels."""
from pathlib import Path
import numpy as np
import pandas as pd
from dlpro.experiment_core import centered_indices,save_json
out=Path('results/classification_label_audit');out.mkdir(exist_ok=True);f=pd.read_csv('data/processed/known_windows.csv');raw=np.load('data/processed/known_rep1.npz')['raw'];fine=np.load('data/processed/fine_rep1.npz')['raw'];f['input100']=[centered_indices(int(v))[0] for v in f.center];f['input200']=[centered_indices(int(v))[1] for v in f.center];f['input50']=[int(np.floor((v+25)/50))-64 for v in f.center]
collisions=[]
for key,g in f.groupby(['input100','input200']):
 if g.type.nunique()<2:continue
 for i in g.index:
  for j in g.index:
   if j<=i or f.type.iloc[i]==f.type.iloc[j]:continue
   collisions.append(dict(first=f.structure_id.iloc[i],first_type=f.type.iloc[i],second=f.structure_id.iloc[j],second_type=f.type.iloc[j],center_difference_bp=int(abs(f.center.iloc[i]-f.center.iloc[j])),raw_two_scale_identical=bool(np.array_equal(raw[i],raw[j])),fine50_identical=bool(np.array_equal(fine[i],fine[j])),same_overlap_group=bool(f.group.iloc[i]==f.group.iloc[j])))
pd.DataFrame(collisions).to_csv(out/'input_collisions.csv',index=False)
rows=[];chin=f[f.type=='CHIN']
for i,row in f[f.type=='CHID'].iterrows():
 j=int(abs(chin.center-row.center).idxmin());other=f.loc[j];rows.append(dict(chid=row.structure_id,nearest_chin=other.structure_id,center_distance_bp=int(abs(row.center-other.center)),chin_within_chid=bool(other.start>=row.start and other.end<=row.end),same_overlap_group=bool(row.group==other.group),two_scale_input_rmse=float(np.sqrt(((raw[i]-raw[j])**2).mean()))))
audit=pd.DataFrame(rows);audit.to_csv(out/'chid_nearest_chin.csv',index=False);save_json(out/'summary.json',dict(n_chid=26,near_chin_centers_100bp=int((audit.center_distance_bp<=100).sum()),near_chin_centers_500bp=int((audit.center_distance_bp<=500).sum()),near_chin_centers_2000bp=int((audit.center_distance_bp<=2000).sum()),different_class_exact_two_scale_collisions=len(collisions),fine_view_collision_count=sum(v['fine50_identical'] for v in collisions),interpretation='Input quantization and nested annotation ambiguity; labels retained exactly, no relabeling or test-derived deletion'))

"""Compare independently localized physical intervals across resolutions."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
from dlpro.local_morphology import interval_iou
p=Path('results/local_contact_morphology');z=np.load(p/'measurements.npz')
k=pd.read_csv('data/processed/known_windows.csv');s=pd.read_csv('data/processed/scan_windows.csv');c=pd.read_csv(p/'localized_candidates.csv')
ids=list(k.structure_id)+[f'scan_{i}' for i in range(len(s))];records=[]
for repeat in (1,2):
 bounds=z[f'bounds_rep{repeat}']
 for i,item in enumerate(ids):
  for left,right in [(0,1),(0,2),(1,2)]:
   a,b=bounds[i,left],bounds[i,right]
   records.append(dict(item_id=item,actual_type=k.type.iloc[i] if i<len(k) else 'unannotated',repeat=repeat,
    scale_a_bp=[50,100,200][left],scale_b_bp=[50,100,200][right],physical_interval_iou=interval_iou(a,b),
    estimated_center_difference_bp=float(abs(a.mean()-b.mean()))))
d=pd.DataFrame(records);d.to_csv(p/'cross_scale_localization.csv',index=False)
rows=[]
for row in c.itertuples():
 part=d[d.item_id==f'scan_{row.scan_index}']
 rows.append(dict(candidate_id=row.candidate_id,minimum_cross_scale_interval_iou=float(part.physical_interval_iou.min()),
  median_cross_scale_interval_iou=float(part.physical_interval_iou.median()),maximum_cross_scale_center_shift_bp=float(part.estimated_center_difference_bp.max()),
  interpretation='Independently localized scales; low overlap can indicate scale-dependent extent or selection of another nearby feature'))
pd.DataFrame(rows).to_csv(p/'candidate_scale_consistency.csv',index=False)
print(pd.DataFrame(rows).drop(columns='interpretation').to_string(index=False))

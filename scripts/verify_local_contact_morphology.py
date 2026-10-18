"""Check local morphology tables against source matrices and recomputed metrics."""
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,f1_score
from dlpro.experiment_core import ContactReader

p=Path('results/local_contact_morphology')
a=pd.read_csv(p/'per_window_geometry.csv');c=pd.read_csv(p/'localized_candidates.csv')
assert len(a)==6420 and len(c)==16
assert not a[['item_id','repeat','bin_bp']].duplicated().any()
assert a.local_support_fraction.between(0,1).all()
for column in ['heatmap_file','comparison_heatmap_file']:
 assert all((p/value).is_file() for value in c[column])
measurements=np.load(p/'measurements.npz')
for name in ['geometry_rep1','geometry_rep2','images_rep1','images_rep2']:
 assert np.isfinite(measurements[name]).all()
pred=pd.read_csv(p/'known_classification_predictions.csv');summary=pd.read_csv(p/'known_classification_summary.csv')
assert pred.groupby(['method','repeat']).structure_id.nunique().eq(344).all()
for (method,repeat),part in pred.groupby(['method','repeat']):
 row=summary[(summary.method==method)&(summary.repeat==repeat)].iloc[0]
 assert abs(accuracy_score(part.actual,part.predicted)-row.accuracy)<1e-12
 assert abs(f1_score(part.actual,part.predicted,average='macro')-row.macro_f1)<1e-12
ns=pd.read_csv(p/'held_type_novelty_scores.csv')
assert ns[ns.actual_type!='background'].groupby(['method','repeat','hidden_type']).item_id.nunique().eq(344).all()
samples=pd.read_csv('data/annotations/samples.csv');checks=[]
for repeat in (1,2):
 reader=ContactReader(samples[(samples.condition=='37C')&(samples.replicate==repeat)].iloc[0].path)
 raw=np.load(f'data/processed/scan_rep{repeat}.npz')['raw']
 for candidate in ['C0075','C0152']:
  row=c[c.candidate_id==candidate].iloc[0];center=int((row.scan_start+row.scan_end)//2)
  start=int(np.floor((center+50)/100)-64)
  difference=float(abs(reader.window(start,128,100)-raw[int(row.scan_index),0]).max())
  assert difference<1e-8
  checks.append(dict(candidate_id=candidate,replicate=repeat,max_raw_matrix_absolute_error=difference))
pd.DataFrame(checks).to_csv(p/'direct_cooler_checks.csv',index=False)
files={str(f.relative_to(p)):{'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()}
 for f in sorted(p.rglob('*')) if f.is_file() and f.name!='integrity.json'}
source_paths=['src/dlpro/local_morphology.py','scripts/analyze_local_contact_morphology.py','scripts/plot_local_contact_morphology.py',
 'scripts/verify_local_contact_morphology.py','scripts/summarize_local_scale_consistency.py','data/processed/known_windows.csv','data/processed/scan_windows.csv',
 'results/unannotated_shape_clusters/candidate_new_structures.csv']
source_hashes={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in source_paths}
report=dict(status='PASS',geometry_rows=len(a),candidate_rows=len(c),localization_pngs=len(c),known_comparison_pngs=len(c),
 classification_rows=len(pred),direct_cooler_checks=len(checks),local_evidence_qualified=int(c.local_evidence_qualified.sum()),sources=source_hashes,files=files)
(p/'integrity.json').write_text(json.dumps(report,indent=2)+'\n')
print({key:value for key,value in report.items() if key not in ['files','sources']})

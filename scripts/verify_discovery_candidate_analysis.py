"""Verify full retention, individual evidence, coordinates and every PNG."""
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
from dlpro.candidate_evidence import assess_inventory
from dlpro.coverage_support import coarse_support
from PIL import Image
p=Path('results/discovery_candidate_analysis');source=pd.read_csv('results/discovery/candidates.csv');f=pd.read_csv(p/'candidate_inventory.csv');clusters=pd.read_csv(p/'cluster_inventory.csv')
assert set(f.candidate_id)==set(source.candidate_id) and len(f)==len(source)==294
assert not f.candidate_id.duplicated().any()
assert f.all_candidates_retained.all()
assert clusters.analysis_eligible.all() and not clusters.minimum_members_gate_used.any() and not clusters.known_member_veto_used.any()
assert len(f[f.quality_pass])==len(pd.read_csv(p/'quality_supported_candidates.csv'))
assert (f.quality_pass&~f.known_buffer_overlap).sum()==len(pd.read_csv(p/'unannotated_candidates.csv'))
expected=assess_inventory(f)
assert expected.tentative_new_shape.eq(f.tentative_new_shape).all()
for row in f.itertuples():
 parts=json.loads(row.genomic_segments)
 assert sum(b-a for a,b in parts)==row.length_bp
 assert parts[0][0]==row.start and parts[-1][1]==row.end
 assert (p/row.heatmap_file).is_file()
 with Image.open(p/row.heatmap_file) as image:image.verify()
for row in clusters.itertuples():
 part=f[f.cluster==row.cluster]
 assert row.members==len(part) and set(row.member_ids.split(';'))==set(part.candidate_id)
 assert row.known_associated_members==int(part.known_buffer_overlap.sum())
 assert row.quality_unannotated_members==int((part.quality_pass&~part.known_buffer_overlap).sum())
scan=pd.read_csv('data/processed/scan_windows.csv');centers=scan.center.to_numpy()
raw={r:np.load(f'data/processed/scan_rep{r}.npz')['raw'][:,0] for r in (1,2)}
support={r:coarse_support(np.load(f'data/processed/coverage_rep{r}.npz')['coverage'],centers)[:,0]>=.8 for r in (1,2)}
expect=np.asarray(json.loads(Path('results/local_contact_morphology/distance_expectation.json').read_text())['100'])
d=abs(np.arange(128)[:,None]-np.arange(128)[None,:]);triangle=np.triu(np.ones((128,128),bool),2)
errors=[]
for row in f.itertuples():
 i=row.scan_index;mask=triangle&support[1][i,:,None]&support[1][i,None,:]&support[2][i,:,None]&support[2][i,None,:]
 a=(raw[1][i]/expect[d])[mask];b=(raw[2][i]/expect[d])[mask]
 corr=float(np.corrcoef(a,b)[0,1]) if len(a)>3 and a.std()>0 and b.std()>0 else np.nan
 if np.isfinite(corr):errors.append(abs(corr-row.repeat_oe_correlation))
 assert abs(mask.sum()/triangle.sum()-row.common_supported_triangle_fraction)<1e-12
assert max(errors)<1e-12
report=dict(status='PASS',retained_candidates=len(f),candidate_pngs=len(list((p/'heatmaps').glob('*.png'))),clusters=len(clusters),
 quality_supported=int(f.quality_pass.sum()),all_known_and_mixed_members_retained=True,maximum_repeat_correlation_error=max(errors),
 files={str(path.relative_to(p)):{'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()} for path in sorted(p.rglob('*')) if path.is_file() and path.name!='integrity.json'},
 sources={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in ['scripts/analyze_discovery_candidates.py','scripts/verify_discovery_candidate_analysis.py','src/dlpro/candidate_evidence.py','results/discovery/candidates.csv','results/discovery/embedding.npz','results/local_contact_morphology/measurements.npz']})
(p/'integrity.json').write_text(json.dumps(report,indent=2)+'\n');print({k:v for k,v in report.items() if k not in ['files','sources']})

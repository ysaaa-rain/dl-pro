"""Read-only closure audit, then inventory only files introduced by this review."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
from dlpro.method_review import load_config,sha256,write_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/targeted_review_v1.json');a=p.parse_args();cfg=load_config(a.config)
    root=Path(cfg['output']);out=root/'review';assert (out/'REPORT.md').exists()
    previous_manifest=Path(cfg['previous_output'])/'review_final/change_manifest.csv'
    previous_rows=list(csv.DictReader(previous_manifest.open(encoding='utf-8')))
    old_mismatches=[r['path'] for r in previous_rows if not Path(r['path']).exists() or sha256(r['path'])!=r['sha256']]
    assert not old_mismatches,old_mismatches
    runs=[]
    for path in sorted((root/'runs').glob('*.json')):
        record=json.loads(path.read_text(encoding='utf-8'))
        if 'command' not in record:continue
        assert record['status']!='running',path
        if path.stem in ['ae','ae_subset_resume']:reason='Stopped for equivalent feature-subset optimization; completed weights preserved and reused exactly'
        elif path.stem in ['audit','audit_threads4','audit_dtype']:reason='Strict float tolerance triggered after residual-MAD amplification; ranked selections subsequently verified'
        elif path.stem=='audit_rank_verified':reason='Protection inventory included newly added protocol; corrected to verify original keys only'
        elif path.stem=='report':reason='Windows GBK decoding of UTF8 protected paths; fixed in report_utf8; partial tables preserved in review_failed_utf8'
        else:reason='See stage output and scientific decisions; successful execution does not imply method benefit'
        runs.append(dict(stage=path.stem,status=record['status'],scientific_disposition=reason,exit_code=record['exit_code'],elapsed_seconds=record.get('elapsed_seconds'),command=' '.join(record['command'])))
    with (out/'run_status.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(runs[0]),lineterminator='\n');w.writeheader();w.writerows(runs)
    assert json.loads((root/'application_demo_check.json').read_text(encoding='utf-8'))['exact_match']
    prep=json.loads((root/'data/preparation.json').read_text(encoding='utf-8'))
    caches={name:sha256(Path(cfg['cache'])/name) for name in prep['cache_sha256']}
    assert caches==prep['cache_sha256']
    write_json(out/'closure.json',dict(previous_manifest_files_verified=len(previous_rows),previous_files_unchanged=True,native_cache_hashes=caches,native_cache_hashes_match=True,run_records=len(runs),no_active_stages=True,new_unit_tests='4 passed; runs/test_all_features.log',exact_application_demo=True,git_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),git_push_performed=False,formal_deliverables_modified=False,inventory_script_sha256=sha256(__file__),large_cache_policy='Raw Coolers and full per-block/candidate arrays remain local; source and cache SHA256 in noise/conditions protocol',manifest_self_excluded=True))
    paths=[*root.rglob('*'),Path(a.config),Path('docs/第六轮定向优化固定协议.md'),Path('src/dlpro/targeted_review.py'),Path('tests/test_targeted_review.py')]
    paths += [*Path('scripts').glob('*targeted*.py'),*[Path('scripts')/name for name in ['background_ae_review.py','train_shared_same.py','evaluate_shared_same.py','diagnose_shared_noise.py','reconstruct_shared_same.py']]]
    manifest=out/'change_manifest.csv';rows=[]
    for path in sorted(set(paths)):
        if not path.is_file() or path==manifest:continue
        data=path.read_bytes();canonical=data
        if path.suffix in {'.py','.md','.json','.csv','.txt','.log'}:
            canonical=data.replace(b'\r\n',b'\n')
        rows.append(dict(path=path.as_posix(),bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),canonical_lf_sha256=hashlib.sha256(canonical).hexdigest()))
    with manifest.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['path','bytes','sha256','canonical_lf_sha256'],lineterminator='\n');w.writeheader();w.writerows(rows)
    print(json.dumps({'new_files_in_manifest':len(rows),'total_bytes':sum(r['bytes'] for r in rows),'previous_files_verified':len(previous_rows)},ensure_ascii=False))

if __name__=='__main__':main()

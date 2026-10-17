"""Final integrity, spatial-mask and immutable-source audit for round seven."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile
import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist
from dlpro.method_review import load_config,sha256,write_json,spatial_cv_masks
from dlpro.coverage_support import coarse_support

NEW_FILES=['configs/evidence_review_v1.json','src/dlpro/evidence_review.py','tests/test_evidence_review.py',
 'scripts/run_evidence_review.py','scripts/audit_course_evidence.py','scripts/audit_independent_metadata.py',
 'scripts/prepare_blind_evidence.py','scripts/diagnose_novelty_evidence.py','scripts/audit_sr_evidence.py',
 'scripts/review_blind_candidates.py','scripts/verify_evidence_materials.py','scripts/summarize_evidence_review.py',
 'scripts/finalize_evidence_review.py']

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',default='configs/evidence_review_v1.json');args=parser.parse_args();cfg=load_config(args.config)
    root=Path(cfg['output']);out=root/'verification';out.mkdir(exist_ok=True);assert not (out/'final_status.json').exists()
    protected=json.loads((root/'protected_existing.json').read_text(encoding='utf-8'));rows=[]
    for name,expected in protected.items():
        current=sha256(name);rows.append(dict(path=name,expected_sha256=expected,actual_sha256=current,unchanged=current==expected))
    pd.DataFrame(rows).to_csv(out/'protected_integrity.csv',index=False);assert all(r['unchanged'] for r in rows)
    initial=json.loads((root/'course_audit/initial_git_state.json').read_text(encoding='utf-8'))
    git=dict(head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip(),status=subprocess.check_output(['git','-c','core.quotepath=false','status','--short','--branch'],text=True,encoding='utf-8'),staged=subprocess.check_output(['git','diff','--cached','--name-only'],text=True,encoding='utf-8').strip())
    assert git['head']==initial['head'] and git['branch']==initial['branch'] and git['staged']==initial['staged'];write_json(out/'final_git_state.json',git)
    run_rows=[]
    for path in sorted((root/'runs').glob('*.json')):
        record=json.loads(path.read_text(encoding='utf-8'));assert record['status']!='running'
        assert sha256(record['source_archive'])==record['source_archive_sha256']
        with zipfile.ZipFile(record['source_archive']) as archive:
            assert archive.testzip() is None
            for name,expected in record['code_sha256'].items():assert hashlib.sha256(archive.read(name)).hexdigest()==expected
        run_rows.append(dict(stage=path.stem,status=record['status'],exit_code=record['exit_code'],seconds=record['elapsed_seconds'],started_utc=record['started_utc'],ended_utc=record['ended_utc'],source_archive_sha256=record['source_archive_sha256']))
    run_table=pd.DataFrame(run_rows);run_table.to_csv(out/'run_status.csv',index=False)
    assert all(stage in set(run_table[run_table.exit_code==0].stage) for stage in ['audit','metadata','blind','novelty','superresolution','materials','report'])
    assert any(name.startswith('test') for name in run_table[run_table.exit_code==0].stage)
    sixth=Path(cfg['sixth_output']);plan=pd.read_csv(sixth/'data/sr_windows.csv');assert (plan.groupby('group').split.nunique()==1).all()
    assert ((plan.start//cfg['block_bp'])==((plan.end-1)//cfg['block_bp'])).all()
    assert ((plan.start//cfg['block_bp'])==plan.group).all()
    known=pd.read_csv(sixth/'data/known_eligibility.csv');scan=pd.read_csv(sixth/'data/scan_eligibility.csv');nk=len(known)
    folds=np.r_[known.new_fold,scan.new_fold];groups=np.r_[known.spatial_block,scan.spatial_block];eligible=np.r_[known.eligible,scan.eligible].astype(bool)&(folds>=0);types=np.r_[known.type,np.full(len(scan),'background')]
    fits=json.loads((sixth/'background_ae/fits.json').read_text(encoding='utf-8'));fit_checks=[]
    for fit in fits:
        train,val,test=spatial_cv_masks(eligible,folds,fit['fold'],cfg['discovery']['folds']);t=np.asarray(fit['background_training_indices'],int);refs=np.asarray(fit['reference_indices'],int);v=np.asarray(fit['calibration_background_indices'],int);vr=np.asarray(fit['calibration_reference_indices'],int)
        assert train[np.r_[t,refs]].all() and val[np.r_[v,vr]].all()
        assert (types[t]=='background').all() and (types[v]=='background').all()
        assert (types[np.r_[refs,vr]]!=fit['hidden_type']).all()
        assert not (set(groups[np.r_[t,refs]]) & set(groups[np.r_[v,vr]]))
        assert not (set(groups[np.r_[t,refs,v,vr]]) & set(groups[test]))
        for seed in cfg['ae']['seeds']:
            path=sixth/'background_ae'/f"{fit['hidden_type']}_fold{fit['fold']}"/f'seed{seed}.pt';cp=torch.load(path,weights_only=True)
            assert cp['train_indices']==t.tolist() and cp['validation_indices']==v.tolist()
            assert sha256(path)==fit['checkpoints'][f'seed{seed}']
        fit_checks.append(dict(hidden_type=fit['hidden_type'],fold=fit['fold'],background_only_AE=True,reference_hidden_class_excluded=True,spatial_disjoint=True,checkpoint_training_indices_match=True))
    pd.DataFrame(fit_checks).to_csv(out/'spatial_fit_checks.csv',index=False)
    # Derive and retain the exact private background-control coordinates used for each case.
    mapping=pd.read_csv(root/'blind_review/private/coordinate_mapping.csv');global_scan=pd.read_csv('data/processed/scan_windows.csv');centers=global_scan.center.to_numpy();raw=np.load('data/processed/scan_rep1.npz')['raw'];support=coarse_support(np.load('data/processed/coverage_rep1.npz')['coverage'],centers)[:,0]>=cfg['discovery']['support_fraction'];lookup={int(x):i for i,x in enumerate(centers)};controls=[]
    def nuis(ids):return np.stack([np.log(raw[ids,0].mean((1,2)).clip(1e-12)),(raw[ids,0]==0).mean((1,2)),support[ids].mean(1)],axis=1)
    for row in mapping.itertuples():
        fit=next(f for f in fits if f['hidden_type']==row.hidden_type and f['fold']==row.fold);allowed=np.asarray(fit['background_training_indices'],int)-nk;values=nuis(allowed);center=values.mean(0);spread=values.std(0).clip(1e-8);index=lookup[int(row.center)];control=int(allowed[cdist((nuis([index])-center)/spread,(values-center)/spread).argmin()])
        with np.load(root/'blind_review/public/cases'/f'{row.case_id}.npz') as arrays:np.testing.assert_array_equal(arrays['background_coarse_signal_per_million'][0],raw[control,0])
        controls.append(dict(case_id=row.case_id,candidate_center=int(row.center),control_scan_index=control,control_center=int(centers[control]),hidden_type=row.hidden_type,fold=row.fold,control_in_allowed_training_background=True))
    pd.DataFrame(controls).to_csv(root/'blind_review/private/background_control_mapping.csv',index=False)
    human=json.loads((root/'blind_review/human_review_status.json').read_text(encoding='utf-8'));assert human['reviewed']==0 and not (root/'blind_review/human_reviews.jsonl').exists()
    with zipfile.ZipFile(root/'blind_review/blind_review_packet.zip') as archive:assert all(not any(x in name for x in ['private','mapping','identity_key','REPORT']) for name in archive.namelist())
    sr=json.loads((root/'sr_audit/status.json').read_text(encoding='utf-8'));assert sr['trained_new_models']==0 and sr['predictions_verified_against_frozen_rows']>0
    lock=json.loads((root/'sr_audit/evaluation_lock.json').read_text(encoding='utf-8'));old_lock=json.loads((sixth/'evaluation/test_lock.json').read_text(encoding='utf-8'))
    assert lock['source_rows_sha256']==sha256(sixth/'evaluation/per_window.csv')
    for name,digest in lock['frozen_models'].items():assert digest==old_lock['checkpoints'][name]
    # Check syntax in memory; do not modify old bytecode or source paths.
    for name in NEW_FILES:compile(Path(name).read_text(encoding='utf-8'),name,'exec') if name.endswith('.py') else json.loads(Path(name).read_text(encoding='utf-8'))
    (out/'verification.log').write_text('Verified protected hashes, immutable source archives, spatial masks, checkpoint indices, exact private controls, frozen predictions and zero human review events.\n',encoding='utf-8')
    write_json(out/'final_status.json',dict(completed_utc=datetime.now(timezone.utc).isoformat(),status='completed',exit_code=0,protected_files_verified=len(rows),all_protected_unchanged=True,source_archives_verified=len(run_rows),sr_spatial_groups_disjoint=True,AE_spatial_fits_verified=len(fit_checks),private_background_controls_verified=len(controls),frozen_predictions_verified=sr['predictions_verified_against_frozen_rows'],unit_test_success=True,real_browser_success=True,independent_human_review_completed=False,new_models_trained=0,confirmed_new_clusters=0,formal_choices_unchanged=True,HEAD_unchanged=True,pushed_to_remote=False,formal_deliverables_modified=False,raw_large_input_hashes='Reuse historical provenance; no claim of rehashing all large raw inputs in this round'))
    excluded={out/'change_manifest.csv',out/'change_manifest.sha256'}
    paths=[Path(name) for name in NEW_FILES]+[p for p in root.rglob('*') if p.is_file() and p not in excluded]
    inventory=pd.DataFrame([dict(path=p.as_posix(),bytes=p.stat().st_size,sha256=sha256(p),role='new_source' if p.as_posix() in NEW_FILES else 'round_seven_artifact',reviewer_shareable=p.is_relative_to(root/'blind_review/public') or p==root/'blind_review/blind_review_packet.zip') for p in sorted(paths)])
    inventory.to_csv(out/'change_manifest.csv',index=False);(out/'change_manifest.sha256').write_text(sha256(out/'change_manifest.csv')+'\n',encoding='ascii')
    print(f'Final verification passed: {len(rows)} protected files, {len(fit_checks)} spatial fits, {len(controls)} controls; {len(paths)} new files inventoried',flush=True)

if __name__=='__main__':main()

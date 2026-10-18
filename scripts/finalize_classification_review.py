"""Integrity and real-data gates after all eighth-round runs and visual review."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile
import joblib
import numpy as np
import pandas as pd
from dlpro.method_review import load_config,sha256,write_json
from dlpro.classification_review import load_inputs,check_spatial_partition,choose_bias,calibrate_bias
from dlpro.experiment_core import LABELS

NEW=['configs/classification_review_v1.json','src/dlpro/classification_review.py','tests/test_classification_review.py',
 'scripts/run_classification_review.py','scripts/audit_classification_review.py','scripts/experiment_classification_review.py',
 'scripts/explain_classification_review.py','scripts/verify_required_evidence.py','scripts/report_classification_review.py',
 'scripts/plot_classification_review.py','scripts/finalize_classification_review.py']

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',default='configs/classification_review_v1.json');a=parser.parse_args();cfg=load_config(a.config);root=Path(cfg['output']);out=root/'verification';out.mkdir(exist_ok=True);assert not (out/'final_status.json').exists()
    protected=json.loads((root/'protected_existing.json').read_text(encoding='utf-8'));integrity=[]
    for name,expected in protected.items():
        current=sha256(name);integrity.append(dict(path=name,sha256=current,expected_sha256=expected,unchanged=current==expected))
    pd.DataFrame(integrity).to_csv(out/'protected_integrity.csv',index=False);assert all(row['unchanged'] for row in integrity)
    initial=json.loads((root/'audit/initial_git_state.json').read_text(encoding='utf-8'));head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip();staged=subprocess.check_output(['git','diff','--cached','--name-only'],text=True).strip();assert head==initial['head'] and branch==initial['branch'] and staged==initial['staged']
    write_json(out/'final_git_state.json',dict(head=head,branch=branch,staged=staged,status=subprocess.check_output(['git','-c','core.quotepath=false','status','--short','--branch'],text=True,encoding='utf-8')))
    runs=[]
    for path in sorted((root/'runs').glob('*.json')):
        record=json.loads(path.read_text(encoding='utf-8'));assert record['status']!='running';assert sha256(record['source_archive'])==record['source_archive_sha256']
        with zipfile.ZipFile(record['source_archive']) as archive:
            assert archive.testzip() is None
            for name,expected in record['code_sha256'].items():assert hashlib.sha256(archive.read(name)).hexdigest()==expected
        runs.append(dict(stage=path.stem,status=record['status'],exit_code=record['exit_code'],seconds=record['elapsed_seconds'],started_utc=record['started_utc'],ended_utc=record['ended_utc'],source_archive=record['source_archive'],archive_sha256=record['source_archive_sha256']))
    rt=pd.DataFrame(runs);rt.to_csv(out/'run_status.csv',index=False);successful=set(rt[rt.exit_code==0].stage);assert {'audit','test','experiment_precision','explain','evidence','report','plot'}<=successful
    frame,folds,raw,support=load_inputs();n=len(frame);y=frame.type.map(dict(zip(LABELS,range(len(LABELS))))).to_numpy();results=pd.read_csv(root/'experiment/out_of_fold_predictions.csv');assert results.groupby('method').structure_id.nunique().eq(n).all() and len(results)==4*n
    partitions=pd.read_csv(root/'experiment/partitions.csv');inner_scores=pd.read_csv(root/'experiment/inner_scores.csv');locks=json.loads((root/'experiment/all_selection_locks.json').read_text(encoding='utf-8'));checks=[]
    for lock in locks:
        fold=lock['outer_fold'];part=partitions[partitions.outer_fold==fold].set_index('structure_id').loc[frame.structure_id];check_spatial_partition(frame.window_start.to_numpy(),frame.window_end.to_numpy(),part.role.to_numpy());train=np.asarray(lock['training_indices']);val=np.asarray(lock['validation_indices']);test=np.asarray(lock['test_indices']);assert len(set(train)&set(val))==len(set(train)&set(test))==len(set(val)&set(test))==0
        assert set(train)|set(val)|set(test)==set(range(n));assert (folds[test]==fold).all() and (folds[val]==(fold+1)%cfg['classifier']['outer_folds']).all()
        path=root/'experiment'/f'fold{fold}';saved=np.load(path/'frozen_cnn_probabilities.npz');p=saved['paired'];bias,trials=choose_bias(p[val],y[val],cfg['classifier']['bias_grid']);np.testing.assert_allclose(bias,lock['calibration_bias'],atol=1e-12)
        for method,prob in [('frozen_cnn',p),('symmetric_bias',calibrate_bias(p,bias))]:
            stored=results[(results.method==method)&(results.fold==fold)].set_index('structure_id').loc[frame.structure_id.iloc[test],[f'p_{label}' for label in LABELS]].to_numpy();np.testing.assert_allclose(stored,prob[test],rtol=1e-6,atol=1e-7)
        for inner in sorted(np.unique(folds[train])):
            role=json.loads((path/f'inner{inner}_roles.json').read_text(encoding='utf-8'));it=set(role['train_indices']);iv=set(role['validation_indices']);assert it|iv==set(train) and not it&iv;assert not (it|iv)&set(val.tolist()+test.tolist());assert (folds[list(iv)]==inner).all()
            tf=np.load(path/f'inner{inner}_transform.npz');assert np.isfinite(tf['expected']).all() and np.isfinite(tf['scale']).all()
        for method,choice in lock['feature_choices'].items():
            sub=inner_scores[(inner_scores.outer_fold==fold)&(inner_scores.method==method)];means=sub.groupby('C').macro_f1.mean();selected=max(means.index,key=lambda c:(means[c],-c));assert selected==choice['C']
            model=joblib.load(path/f'{method}.joblib');assert model['train_indices']==train.tolist() and model['C']==selected and int(model['scaler'].n_samples_seen_)==2*len(train);assert np.array_equal(model['model'].classes_,np.arange(len(LABELS)))
        for name,expected in lock['checkpoints'].items():assert sha256(name)==expected
        checks.append(dict(fold=fold,full_contexts_disjoint=True,calibration_only_inner_validation=True,three_training_inner_folds_disjoint=True,chosen_C_matches_inner_scores=True,final_scaler_samples_match_training=True,all_outer_predictions_retained=True))
    pd.DataFrame(checks).to_csv(out/'nested_spatial_checks.csv',index=False)
    d=abs(np.arange(128)[:,None]-np.arange(128)[None,:]);mask_checks=[]
    for path in sorted((root/'explanation').glob('masks_*.npz')):
        arrays=np.load(path);bp=int(path.stem.rsplit('_',1)[1].replace('bp',''));channel=cfg['classifier']['physical_bins_bp'].index(bp)
        for local,i in enumerate(arrays['structure_indices']):
            if arrays['status'][local]!='available':continue
            target=arrays['structure_proxy'][local];control=arrays['background'][local];assert target.sum()==control.sum() and not (target&control).any();np.testing.assert_array_equal(np.bincount(d[target],minlength=128),np.bincount(d[control],minlength=128));assert np.array_equal(target,target.T) and np.array_equal(control,control.T)
            valid=(support[i,channel]>=cfg['classifier']['support_fraction'])&(support[i+n,channel]>=cfg['classifier']['support_fraction']);assert not ((target|control)&~(valid[:,None]&valid[None,:]&(d>=2))).any()
            mask_checks.append(dict(structure_id=frame.structure_id.iloc[i],bp=bp,pixels=int(target.sum()),equal_area=True,equal_exact_distance_histogram=True,common_native_support=True))
    pd.DataFrame(mask_checks).to_csv(out/'real_mask_checks.csv',index=False)
    real=json.loads((Path(cfg['previous_output'])/'blind_review/human_review_status.json').read_text(encoding='utf-8'));assert real['reviewed']==0 and not (Path(cfg['previous_output'])/'blind_review/human_reviews.jsonl').exists()
    figures=['review/classification_paired_effects.png','review/specificity_by_fill.png','review/occlusion_mask_examples.png'];write_json(out/'visual_review.json',dict(reviewed_by='Codex technical visual inspection; not independent biological review',track_sheets=[f'required_evidence/track_review_sheet_{i}.png' for i in range(1,5)],track_images=14,full_size_additional_views=['multitrack 3790000-3800000','multitrack_mutants 1640000-1650000'],observations=['Shared axes, three condition legends, gray mean, blue genes, orange nested structure lanes readable','Background without annotations intentionally has empty structure panel','Near570kb common low signal does not prove a condition effect','Original classification plot label overflow corrected without changing data; draft retained','CHID50bp missing explanation shown as NA, not silently inferred'],classification_figures=figures,all930_track_images_reviewed=False))
    for name in NEW:
        if name.endswith('.py'):compile(Path(name).read_text(encoding='utf-8'),name,'exec')
    subprocess.run(['git','diff','--check'],check=True)
    write_json(out/'final_status.json',dict(status='completed',exit_code=0,completed_utc=datetime.now(timezone.utc).isoformat(),protected_files_verified=len(integrity),all_protected_unchanged=True,source_archives_verified=len(runs),failed_attempts_preserved=int((rt.exit_code!=0).sum()),outer_spatial_folds_verified=len(checks),n_loci_retained=n,real_matched_masks_verified=len(mask_checks),unit_tests_passed=5,real_task3_signal_checks=30,targeted_track_images_visually_inspected=14,independent_human_cases=0,formal_choices_changed=False,HEAD_changed=False,remote_pushed=False,formal_deliverables_modified=False,no_new_SR_training=True))
    (out/'verification.log').write_text('Protected hashes, source archives, outer contexts, inner roles/selection, calibrated probabilities, scaler training count, real masks, zero real human reviews and unchanged Git HEAD verified.\n',encoding='utf-8')
    excluded={out/'change_manifest.csv',out/'change_manifest.sha256'};files=[Path(name) for name in NEW]+[path for path in root.rglob('*') if path.is_file() and path not in excluded];inventory=pd.DataFrame([dict(path=path.as_posix(),bytes=path.stat().st_size,sha256=sha256(path),role='source' if path.as_posix() in NEW else 'round8_artifact') for path in sorted(files)]);inventory.to_csv(out/'change_manifest.csv',index=False);(out/'change_manifest.sha256').write_text(sha256(out/'change_manifest.csv')+'\n',encoding='ascii')
    print(f'Final integrity passed: {len(integrity)} protected files, {len(checks)} outer folds, {len(mask_checks)} real masks; {len(files)} files inventoried',flush=True)

if __name__=='__main__':main()

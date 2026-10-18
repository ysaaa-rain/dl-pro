"""Reproduce the actual historical AE and freeze all pre-existing work."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
import pandas as pd
import torch
from dlpro.experiment_core import Autoencoder, ShapeTransform, LENGTH, physical_intervals, configure
from dlpro.method_review import load_config, sha256, write_json, nms

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args()
    cfg=load_config(a.config);out=Path(cfg['output']);old=Path(cfg['previous_output'])
    assert not (out/'audit.json').exists()
    (out/'data').mkdir(exist_ok=True)
    for source in (old/'data').iterdir():
        if source.is_file():
            dest=out/'data'/source.name
            if not dest.exists(): shutil.copyfile(source,dest)
            assert sha256(source)==sha256(dest)
    files=set(subprocess.check_output(['git','ls-files','-z']).decode('utf-8').split('\0'))
    files.update(subprocess.check_output(['git','ls-files','--others','--exclude-standard','-z']).decode('utf-8').split('\0'))
    protected={s:sha256(s) for s in sorted(files) if s and Path(s).is_file() and not s.startswith((cfg['output']+'/', 'scripts/', 'src/', 'tests/', 'configs/'))}
    if (out/'protected_existing_files.json').exists():
        original=json.loads((out/'protected_existing_files.json').read_text(encoding='utf-8'))
        assert all(Path(name).is_file() and sha256(name)==value for name,value in original.items())
    else: write_json(out/'protected_existing_files.json',protected)
    raw=np.load('data/processed/scan_rep1.npz')['raw']
    # Historical configure() uses four CPU threads; preserve its floating point route.
    configure(42);rows=[];begin=time.monotonic()
    for fold in range(cfg['discovery']['folds']):
        directory=Path('results/discovery_spatial_cv')/f'fold{fold}'
        model=Autoencoder().eval();cp=torch.load(directory/'ae.pt',weights_only=True);model.load_state_dict(cp['state_dict'])
        tf=ShapeTransform.load(directory/'transform.npz');x=tf.transform(raw)[:,:1]
        with torch.no_grad(): rec=np.concatenate([model(b).numpy() for b in torch.from_numpy(x).split(64)])
        error=np.log(np.maximum(((rec-x)**2).mean((1,2,3)),1e-12))
        density=np.log(np.maximum(raw[:,0].mean((1,2)),1e-12))
        null=np.load(directory/'ae_null.npz');bucket=np.searchsorted(null['edges'],density)
        # sklearn IsotonicRegression.predict casts its interpolation to input dtype.
        prediction=np.interp(density,null['density_thresholds'],null['error_thresholds']).astype(density.dtype)
        score=(error-prediction-null['location'][bucket])/null['spread'][bucket]
        stored=pd.read_csv(directory/'all_scores.csv').noise_adjusted_ae.to_numpy()
        # A near-zero calibrated MAD amplifies float32 inference drift. Verify
        # numerical closeness AND the actual fixed-budget ranked selections.
        difference=float(abs(score-stored).max());assert np.allclose(score,stored,atol=1e-3,rtol=1e-6),(fold,difference)
        scan_centers=pd.read_csv('data/processed/scan_windows.csv').center.to_numpy()
        for budget in cfg['discovery']['global_budgets']:
            np.testing.assert_array_equal(nms(score,scan_centers,budget,cfg['discovery']['nms_bp'],LENGTH),nms(stored,scan_centers,budget,cfg['discovery']['nms_bp'],LENGTH))
        history=pd.read_csv(directory/'ae_history.csv')
        ids=np.load(directory/'features.npz')
        rows.append(dict(fold=fold,selected_epoch=int(cp['selected_epoch']),history_epochs=len(history),train_background=int(ids['background_train'].sum()),validation_background=int(ids['background_val'].sum()),score_max_abs_diff=difference,weights_sha256=sha256(directory/'ae.pt'),train_final=float(history.train_mse.iloc[-1]),validation_final=float(history.val_mse.iloc[-1])))
    pd.DataFrame(rows).to_csv(out/'historical_ae_reproduction.csv',index=False)
    scan=pd.read_csv('data/processed/scan_windows.csv')
    intervals=sorted(interval for c in scan.center for interval in physical_intervals(c,cfg['discovery']['max_context_bp']))
    merged=[]
    for start,end in intervals:
        if merged and start<=merged[-1][1]: merged[-1][1]=max(merged[-1][1],end)
        else: merged.append([start,end])
    coverage=sum(b-a for a,b in merged)
    oldsr=json.loads((old/'sr_corrected/training_status.json').read_text(encoding='utf-8'))
    write_json(out/'audit.json',dict(historical_ae_reproduced=True,historical_ae_training_rerun=False,old_scan_covered_bp=coverage,genome_bp=LENGTH,unexposed_spatial_region_established=False,all_results_retrospective=True,previous_shared_training_already_implemented=True,previous_shared_training_source='train_allocation_review.py: rng.integers(0,2) + make_batch',new_question='Matched unique-count same-repeat NLL: repeat1-only versus a shared mapping trained from both biological repeats',same_aggregation='Cooler coarse pixels are area-averaged native image counts; 40bp input equals avg_pool2d(native,4)',prior_source=oldsr,elapsed_seconds=time.monotonic()-begin))
    print('Historical AE reproduced, source work protected; full genome previously scanned',flush=True)

if __name__=='__main__':main()

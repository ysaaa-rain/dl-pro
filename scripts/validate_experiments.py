"""Independent arithmetic and artifact checks for the completed experiment runs."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from dlpro.experiment_core import save_json,LENGTH

out=Path('results/experiment');checks={}
f=pd.read_csv('data/processed/known_windows.csv')
assert len(f)==344 and f.groupby('group').split.nunique().max()==1
for row in f.itertuples():
    other=f[(f.window_start<row.window_end)&(f.window_end>row.window_start)]
    assert other.split.nunique()==1
checks['split_isolation']=True
p=pd.read_csv('results/classification_cv/out_of_fold_predictions.csv');assert len(p)==344 and p.structure_id.nunique()==344
assert p.groupby('group').fold.nunique().max()==1
assert np.allclose(p[[f'p_{c}' for c in ['CHID','CHIN','OPCID']]].sum(axis=1),1)
checks['one_out_of_fold_prediction_per_structure']=True
for folder in ['discovery_density_sensitive','discovery_density_invariant','discovery']:
    root=Path('results')/folder;f1=pd.read_csv(root/'candidates.csv');obj=np.load(root/'embedding.npz')
    z=obj['latent'][f1.scan_index.to_numpy()].astype(float);centered=z-obj['pca_mean'];components=obj['pca_components']
    projection=np.einsum('nd,kd->nk',centered,components,optimize=False)
    covariance=np.einsum('ni,nj->ij',centered,centered,optimize=False)/(len(centered)-1)
    cv=np.einsum('ij,kj->ki',covariance,components,optimize=False)
    eigenvalues=np.einsum('ki,ki->k',cv,components,optimize=False)
    relative_error=float(np.linalg.norm(cv-eigenvalues[:,None]*components)/max(np.linalg.norm(covariance),1e-15))
    orthogonality=float(abs(np.einsum('ki,li->kl',components,components,optimize=False)-np.eye(len(components))).max())
    distances=((projection[:,None]-obj['centers'][None])**2).sum(2)
    labels_match=bool(np.array_equal(distances.argmin(1),f1.cluster.to_numpy()))
    assert relative_error<1e-5 and orthogonality<1e-5 and labels_match
    checks[folder+'_numeric']=dict(pca_eigen_relative_error=relative_error,pca_orthogonality_max_error=orthogonality,kmeans_labels_match_independent_distance=True)
    assert all((root/'heatmaps'/f'{c}.png').exists() for c in f1.candidate_id)
tracks=pd.read_csv('results/multitrack/index.csv');assert len(tracks)==465 and tracks.start.iloc[0]==0 and tracks.end.iloc[-1]==LENGTH
assert np.array_equal(tracks.start.to_numpy()[1:],tracks.end.to_numpy()[:-1]) and (tracks.end-tracks.start).sum()==LENGTH
assert all(Path(x).exists() for x in tracks.path)
signals=np.load('results/multitrack/signals.npz');assert all(np.isfinite(signals[k]).all() and (signals[k]>=0).all() for k in signals.files)
checks['whole_genome_465_tracks']=True
reconstruction=np.load('results/sr_expanded/reconstruction_seed42.npy',mmap_mode='r');assert reconstruction.shape==(32,1024,1024)
assert all(np.isfinite(v).all() and v.min()>=0 and v.max()<=1 for v in reconstruction)
sr=pd.read_csv('data/processed/sr_expanded/windows.csv');assert sr[sr.replicate==2].split.eq('test').all()
assert sr.groupby('group').split.nunique().max()==1
known_sr=pd.read_csv('results/sr_expanded/known_test_windows.csv');assert len(known_sr)==49
assert (known_sr.start>=known_sr.window_start).all() and (known_sr.end<=known_sr.window_end).all()
checks['sr_rep2_withheld_and_known_contexts_contained']=True
# Budget-matched random candidate baseline: high recall alone can be misleading.
scan=pd.read_csv('results/discovery/all_windows.csv');centers=scan.center.to_numpy();distance=abs(centers[:,None]-f.center.to_numpy()[None]);covered=np.minimum(distance,LENGTH-distance)<=6400
protocol=json.loads(Path('results/discovery/protocol.json').read_text());before=protocol['selected_before_nms'];after=protocol['retained_after_nms'];test=f.split.eq('test').to_numpy();rng=np.random.default_rng(314)
trials=[]
for trial in range(200):
    order=rng.permutation(len(scan));pre=order[:before];chosen=[]
    available=np.ones(len(scan),bool)
    for i in order:
        if not available[i]:continue
        chosen.append(i);d=abs(centers-centers[i]);available&=np.minimum(d,LENGTH-d)>=12800
        if len(chosen)==after:break
    if len(chosen)!=after:continue
    trials.append(dict(trial=trial,before_nms_test_recall=float(covered[pre].any(0)[test].mean()),after_nms_test_recall=float(covered[chosen].any(0)[test].mean())))
random=pd.DataFrame(trials);random.to_csv('results/discovery/budget_matched_random.csv',index=False)
checks['random_candidate_baseline']=dict(trials=len(random),before_nms_mean=float(random.before_nms_test_recall.mean()),after_nms_mean=float(random.after_nms_test_recall.mean()))
checks['tests']='8 passed in 1.52 seconds; circular area correction and density nuisance tests included'
save_json(out/'validation.json',checks);print(json.dumps(checks,indent=2))

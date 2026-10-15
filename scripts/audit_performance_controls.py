"""Verify geometry, folds, fixed selections and source code fingerprints."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from dlpro.experiment_core import save_json
out=Path('results/performance_validation');out.mkdir(exist_ok=True);checks={};dense=pd.read_csv('data/processed/sr_dense/windows.csv');old=pd.read_csv('data/processed/sr_expanded/windows.csv');assert len(dense)==352 and dense.original_sparse.sum()==64;assert ((dense.start//100000)==((dense.end-1)//100000)).all();assert not set(dense.group)&set(old[old.split!='train'].group);checks['dense_train_only_352_and_sparse_subset64']=True
known=pd.read_csv('data/processed/known_windows.csv');folds=pd.read_csv('results/classification_cv/out_of_fold_predictions.csv').set_index('structure_id').loc[known.structure_id,'fold'].to_numpy();pool=pd.read_csv('results/classification_repeat_ssl/pretraining_pool.csv')
for fold,g in pool.groupby('fold'):
 held=known[(folds==fold)|(folds==(fold+1)%5)];overlap=(g.start.to_numpy()[:,None]<held.window_end.to_numpy()[None])&(g.end.to_numpy()[:,None]>held.window_start.to_numpy()[None]);assert not overlap.any();d=abs(g.start.to_numpy()[:,None]-g.start.to_numpy()[None]);d=np.minimum(d,4641652-d);assert (d[~np.eye(len(g),dtype=bool)]>=25600).all()
checks['all_ssl_pools_purged_and_nonoverlap']=True;checks['ssl_pool_counts_by_fold']=pool.groupby('fold').size().to_dict();pred=pd.read_csv('results/classification_repeat_ssl/out_of_fold_predictions.csv')
for _,g in pred.groupby('recipe'):
 assert len(g)==g.structure_id.nunique()==344;assert np.array_equal(g.set_index('structure_id').loc[known.structure_id,'fold'].to_numpy(),folds);p=g[['p_CHID','p_CHIN','p_OPCID']].to_numpy();assert np.isfinite(p).all() and np.allclose(p.sum(1),1)
checks['each_classification_recipe_344_unique_oof']=True
sr=json.loads(Path('results/sr_data_controls/selection.json').read_text());cl=json.loads(Path('results/classification_repeat_ssl/decisions.json').read_text());protocol_hash=hashlib.sha256(Path('results/performance_protocol/plan.json').read_bytes()).hexdigest();assert sr['protocol_sha256']==cl['protocol_sha256']==protocol_hash;assert sr['selection_only_validation'] and all(d['selection_only_validation'] for d in cl['folds']);checks['shared_fixed_protocol_and_validation_selections']=True
shape=pd.read_csv('results/sr_data_controls/shape_metrics.csv');assert shape.groupby('method').size().eq(98).all();native=pd.read_csv('results/sr_data_controls/native_metrics.csv');assert native.groupby('method').size().eq(32).all();assert len(set(native.method))==14;checks['all9_sr_models_and5_baselines_complete']=True
sources=list(Path('src/dlpro').glob('*.py'))+[Path('scripts')/p for p in ['train_sr_data_controls.py','evaluate_sr_data_controls.py','train_classifier_repeat_ssl.py','evaluate_classifier_repeat_ssl.py','summarize_performance_controls.py']];save_json(out/'code_sha256.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources});save_json(out/'integrity.json',checks);print(json.dumps(checks,indent=2),flush=True)

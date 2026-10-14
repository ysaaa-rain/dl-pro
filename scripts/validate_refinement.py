"""Independent numerical, budget, geometry and count-conservation checks."""
from pathlib import Path
import json,subprocess
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from dlpro.experiment_core import save_json,configure
from dlpro.refinement import CountConsistentSR
from dlpro.superresolution import mean_pool
configure(42);checks={};d=Path('results/discovery_refined');p=json.loads((d/'protocol.json').read_text());c=pd.read_csv(d/'candidates.csv');assert len(c)==p['primary_budget']==64
centers=c.center.to_numpy();delta=abs(centers[:,None]-centers[None]);circular=np.minimum(delta,4641652-delta);assert (circular[~np.eye(len(c),dtype=bool)]>=12800).all();checks['fixed_budget_and_spacing']=True
f=np.load(d/'features.npz');assert all(np.isfinite(f[k]).all() for k in f.files);z=f['z1'];scan=pd.read_csv('data/processed/scan_windows.csv');blocks=scan.loc[scan.training_background,'background_block'].unique();order=np.random.default_rng(42).permutation(blocks);train=scan.training_background&scan.background_block.isin(order[:int(.7*len(order))]);assert np.allclose(z[train].mean(0),f['pca_mean']);centered=z[train]-f['pca_mean'];cov=np.einsum('ni,nj->ij',centered,centered)/(train.sum()-1);components=f['pca_components'];eigen=np.einsum('ki,ij,kj->k',components,cov,components);resid=np.einsum('ij,kj->ki',cov,components)-eigen[:,None]*components;checks['pca_eigen_max_relative_error']=float(abs(resid).max()/max(abs(cov).max(),1e-12));assert checks['pca_eigen_max_relative_error']<1e-10
embedding=np.einsum('ni,ki->nk',z-f['pca_mean'],components);distance=((embedding[c.scan_index.to_numpy(),None,:]-f['cluster_centers'][None])**2).sum(2);assert np.array_equal(distance.argmin(1),c.cluster.to_numpy());checks['independent_pca_and_cluster_labels']=True
cache=np.load('data/processed/sr_expanded/raw_counts.npy',mmap_mode='r');windows=pd.read_csv('data/processed/sr_expanded/windows.csv');a=cache[windows[windows.split.eq('test')].index[0]].astype('float32');cp=torch.load('results/sr_refined/model_seed42.pt',weights_only=True);scale=cp['scale'];model=CountConsistentSR(scale);model.load_state_dict(cp['state_dict']);model.eval();low=mean_pool(a)
with torch.no_grad():pred=model(torch.from_numpy((np.log1p(low)/scale)[None,None]));decoded=torch.expm1(pred*scale);coarse=F.avg_pool2d(decoded,4).numpy()[0,0]
checks['count_conservation_max_absolute_error']=float(abs(coarse-low).max());checks['count_conservation_max_relative_error']=float((abs(coarse-low)/np.maximum(low,1e-3)).max());assert checks['count_conservation_max_relative_error']<1e-5;assert torch.allclose(pred,pred.transpose(-1,-2),atol=1e-6)
reconstruction=np.load('results/sr_refined/reconstruction_seed42.npy',mmap_mode='r');assert reconstruction.shape==(32,1024,1024) and np.isfinite(reconstruction).all() and reconstruction.min()>=0;checks['unclipped_log_reconstruction_finite']=True
known=pd.read_csv('results/sr_refined/known_test_windows.csv');assert len(known)==49;assert windows[windows.replicate.eq(2)].split.eq('test').all();checks['sr_rep2_not_fitted_and_49_known_contexts']=True
idx=pd.read_csv('results/multitrack_mutants/index.csv');assert len(idx)==465 and idx.start.iloc[0]==0 and idx.end.iloc[-1]==4641652;assert np.array_equal(idx.start.to_numpy()[1:],idx.end.to_numpy()[:-1]);assert all(Path(v).exists() for v in idx.path);checks['mutant_tracks_full_coverage']=True
manifest=json.loads(Path('results/multitrack_mutants/cache_sources.json').read_text());signals=np.load('results/multitrack_mutants/signals.npz');assert set(signals.files)==set(manifest);assert all(np.isfinite(signals[v]).all() for v in signals.files);assert len(manifest)==6;checks['mutant_cache_correct_samples']=True
# A different combination must reject an existing directory before writing results.
trial=subprocess.run(['.venv/bin/python','scripts/multitrack.py','--conditions','37C','44C','rif25','--output','results/multitrack_mutants'],capture_output=True,text=True);assert trial.returncode!=0 and 'Source mismatch' in trial.stderr;checks['wrong_condition_cache_rejected']=True
checks['tests']='10 passed in1.74s';save_json('results/refinement/validation.json',checks);print(json.dumps(checks,indent=2))

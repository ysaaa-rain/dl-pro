"""Independent artifact checks and paired SR uncertainty using100kb spatial blocks."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from dlpro.experiment_core import configure,save_json
from dlpro.refinement import ContextCountSR
from dlpro.superresolution import mean_pool
configure(42);out=Path('results/model_validation');checks={};rng=np.random.default_rng(328)
def interval(frame,group,value):
 groups=frame[group].unique();assert len(groups)>=2
 draws=[]
 for _ in range(2000):
  sampled=pd.concat([frame[frame[group]==g] for g in rng.choice(groups,len(groups),replace=True)])
  draws.append(sampled[value].mean())
 return dict(n=len(frame),n_spatial_blocks=len(groups),mean=float(frame[value].mean()),block95=np.quantile(draws,[.025,.975]).tolist())
windows=pd.read_csv('data/processed/sr_expanded/windows.csv');native=pd.read_csv('results/sr_context/metrics_per_window.csv');old=pd.read_csv('results/sr_refined/projection_ablation_native.csv');native=native[(native.method=='subpixel_cnn')&(native.seed==42)].merge(old[old.method=='projected_first_round_cnn'],on=['sample_index','replicate'],suffixes=('_context','_projection'),validate='one_to_one');native['spatial_block']=windows.loc[native.sample_index,'group'].to_numpy();cis={}
for metric in ['psnr','ssim']:
 native['delta']=native[metric+'_context']-native[metric+'_projection'];cis['context_minus_projected_cnn_native_'+metric]=interval(native,'spatial_block','delta')
shape=pd.read_csv('results/sr_context/known_recovery.csv');shape=shape[shape.method=='subpixel_cnn'];old=pd.read_csv('results/sr_refined/projection_ablation_shape.csv');shape=shape.merge(old[old.method=='projected_first_round_cnn'],on=['structure_id','type','replicate'],validate='one_to_one');known=pd.read_csv('results/sr_context/known_test_windows.csv').set_index('structure_id');shape['spatial_block']=(known.loc[shape.structure_id,'center'].to_numpy()//100000).astype(int);shape['delta']=shape.context_oe_ssim_original_100bp-shape.oe_ssim_100bp;cis['context_minus_projected_cnn_100bp_oe_ssim']=interval(shape,'spatial_block','delta');save_json(out/'sr_block_intervals.json',cis)
cp=torch.load('results/sr_context/model_seed42.pt',weights_only=True);model=ContextCountSR(cp['scale']);model.load_state_dict(cp['state_dict']);model.eval();cache=np.load('data/processed/sr_expanded/raw_counts.npy',mmap_mode='r');idx=windows[windows.split.eq('test')].index[0];low=mean_pool(cache[idx].astype('float32'));x=torch.from_numpy((np.log1p(low)/cp['scale'])[None,None])
with torch.no_grad():pred=model(x);decoded=torch.expm1(pred*cp['scale']);coarse=F.avg_pool2d(decoded,4).numpy()[0,0]
error=float((abs(coarse-low)/np.maximum(low,1e-3)).max());assert error<1e-5;assert torch.allclose(pred,pred.transpose(-1,-2),atol=1e-6);checks['context_sr_count_relative_error']=error
reconstruction=np.load('results/sr_context/reconstruction_seed42.npy',mmap_mode='r');assert reconstruction.shape==(32,1024,1024);assert np.isfinite(reconstruction).all() and reconstruction.min()>=0;checks['context_reconstruction_finite_nonnegative']=True
candidates=pd.read_csv('results/discovery_operating_point/candidates.csv')
for (fold,separation),g in candidates.groupby(['fold','separation_bp']):
 d=abs(g.center.to_numpy()[:,None]-g.center.to_numpy()[None]);d=np.minimum(d,4641652-d);assert (d[~np.eye(len(g),dtype=bool)]>=separation).all()
checks['calibrated_candidate_spacing_all_folds']=True
recall=pd.read_csv('results/discovery_operating_point/out_of_fold_recall.csv')
for _,g in recall.groupby('separation_bp'):assert len(g)==g.structure_id.nunique()==344
checks['344_unique_oof_recall_per_policy']=True
# Eligible backgrounds must exclude every known validation/test maximum input window.
known=pd.read_csv('data/processed/known_windows.csv');folds=pd.read_csv('results/classification_cv/out_of_fold_predictions.csv').set_index('structure_id').loc[known.structure_id,'fold'].to_numpy();scan=pd.read_csv('data/processed/scan_windows.csv')
for fold in range(5):
 p=np.load(f'results/discovery_support_cv/fold{fold}_parameters.npz');held=known.center.to_numpy()[(folds==fold)|(folds==(fold+1)%5)];centers=scan.center.to_numpy()[p['background_train']];d=abs(centers[:,None]-held[None]);assert (np.minimum(d,4641652-d)>=25600).all()
checks['background_train_purged_heldout_maximum_contexts']=True
save_json(out/'extended_checks.json',checks);print(cis,checks,flush=True)

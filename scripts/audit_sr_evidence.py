"""Strong-baseline paired audit and all-count strata using frozen weights only."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from dlpro.method_review import load_config,sha256,write_json
from dlpro.sr_review import load_data,make_batch,infer_model
from dlpro.evidence_review import block_statistics
from evaluate_shared_same import load_models
from evaluate_allocation_review import simple_predictions,interval

def paired_effects(frame,metrics,keys,references,cfg):
    rows=[]
    for group,part in frame.groupby(keys,dropna=False):
        group=group if isinstance(group,tuple) else (group,)
        for method in sorted(part.method.unique()):
            for ref in references:
                if ref==method:continue
                left=part[part.method==method];right=part[part.method==ref]
                pair=left.merge(right,on=['group','start','source_repeat'],suffixes=('_a','_b'),validate='one_to_one')
                for metric in metrics:
                    valid=pair[[metric+'_a',metric+'_b']].notna().all(1);sub=pair[valid]
                    result=interval((sub[metric+'_a']-sub[metric+'_b']).to_numpy(),sub.group,cfg) if len(sub) else dict(delta=None,low=None,high=None,groups=0)
                    rows.append(dict(zip(keys,group),method=method,reference=ref,metric=metric,paired_windows=len(sub),empty_pairs=len(pair)-len(sub),**result))
    return pd.DataFrame(rows)

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();cfg=load_config(a.config);root=Path(cfg['output']);out=root/'sr_audit';out.mkdir(exist_ok=True);assert not (out/'status.json').exists()
    assert (root/'novelty_diagnosis/status.json').exists(), 'Finish independent/novelty audit before SR stage'
    sixth=Path(cfg['sixth_output']);frame=pd.read_csv(sixth/'evaluation/per_window.csv');metrics=['psnr','ssim','raw_mae','raw_nrmse','log_mse','stabilized_poisson_deviance_per_count','oe_50bp_ssim','oe_100bp_ssim','oe_160bp_ssim']
    frame.groupby(['split','comparison','method'])[metrics].mean().to_csv(out/'unified_all_metrics.csv')
    frame.groupby(['split','comparison','source_repeat','method'])[metrics].mean().to_csv(out/'directional_metrics.csv')
    frame.groupby('method').agg(max_block_error=('block_mean_max_relative_error','max'),max_symmetry_error=('symmetry_max_relative_error','max'),minimum_prediction=('prediction_min','min'),constrained=('constrained','all')).to_csv(out/'constraints_by_method.csv')
    mean=frame.groupby(['split','comparison','method','group','start','source_repeat'])[metrics].mean().reset_index()
    references=['train_distance_prior','bicubic_unprojected','bilinear_unprojected','projected_bicubic','uniform','legacy_sparse_mse']
    paired_effects(mean,metrics,['split','comparison'],references,cfg).to_csv(out/'all_paired_intervals.csv',index=False)
    # No method is selected from this audit. Validation choice and operational choice remain frozen.
    original_cfg=load_config('configs/targeted_review_v1.json');plan,raw,depth=load_data(original_cfg);sr=original_cfg['sr'];ec=cfg['evidence'];torch.set_num_threads(cfg['threads'])
    params=np.load(Path(cfg['previous_output'])/'sr_corrected/normalization.npz');scale=float(params['scale']);models,new_hash=load_models(original_cfg,params);old_models,old_hash=load_models(original_cfg,params,old=True);models.update(old_models)
    models={key:model for key,model in models.items() if key[0] in ec['stratified_methods']}
    lock=json.loads((sixth/'evaluation/test_lock.json').read_text(encoding='utf-8'))
    for name,digest in {**old_hash,**new_hash}.items():assert lock['checkpoints'][name]==digest
    write_json(out/'evaluation_lock.json',dict(frozen_models={**old_hash,**new_hash},source_rows_sha256=sha256(sixth/'evaluation/per_window.csv'),count_edges=cfg['diagnostic']['count_edges'],distance_edges_bp=cfg['diagnostic']['distance_edges_bp'],support_edges=ec['support_edges'],new_training=False,strata_used_for_model_selection=False))
    n=sr['patch_bins']//sr['factor'];f=sr['factor'];ix,jx=np.triu_indices(n);d=(jx-ix)*f*sr['native_bp'];unique=np.ones((len(ix),f,f),bool);unique[ix==jx]=np.triu(np.ones((f,f),bool));weights=unique.sum((1,2));rows=[]
    def blocks(image):return image.reshape(n,f,n,f).transpose(0,2,1,3)[ix,jx]
    frozen_lookup=frame.set_index(['start','source_repeat','method','seed','comparison']); verified_predictions=0
    with torch.no_grad():
        for index,row in plan[plan.split.isin(['new_test','historical_test'])].iterrows():
            highs=[make_batch(raw,depth,[index],[r])[0] for r in (0,1)];b=[blocks(x.numpy()[0,0]) for x in highs];masses=[x.sum((1,2)) for x in b];pair_mass=(masses[0]+masses[1])/2
            valid_fraction=(((b[0]>0).astype(float)+(b[1]>0))*unique).sum((1,2))/(2*weights)
            masks=[]
            for lower,upper in zip(cfg['diagnostic']['distance_edges_bp'][:-1],cfg['diagnostic']['distance_edges_bp'][1:]):
                band=(d>=lower)&(d<upper)
                for axis,edges,values in [('count',cfg['diagnostic']['count_edges'],pair_mass),('effective_pixels',ec['support_edges'],valid_fraction)]:
                    for lo,hi in zip(edges[:-1],edges[1:]):masks.append((axis,lower,upper,lo,hi,band&(values>=lo)&(values<hi)))
            for r in (0,1):
                high=highs[r];low=F.avg_pool2d(high,f);predictions={(name,-1):pred for name,pred in simple_predictions(low,params['profile'],f).items()}
                for key,model in models.items():predictions[key]=infer_model(model,key[0],low,scale,sr)[0]
                for (method,seed),pred in predictions.items():
                    assert torch.isfinite(pred).all() and float(pred.min())>=0
                    assert float((pred-pred.transpose(-1,-2)).abs().max()/pred.max().clamp_min(1e-8))<1e-5
                    if method not in ['bilinear_unprojected','bicubic_unprojected']:
                        assert float(((F.avg_pool2d(pred,f)-low).abs()/low.clamp_min(1e-8)).max())<1e-5
                    pb=blocks(pred.numpy()[0,0]);logp=(np.log1p(pb)/scale).clip(0,1)
                    for comparison,target in [('same_repeat',r),('cross_repeat',1-r)]:
                        statistics=block_statistics(pb,b[target],b[r],unique);logerror=((logp-(np.log1p(b[target])/scale).clip(0,1))**2*unique).sum((1,2))/weights
                        actual=float(np.average(statistics['raw_mae'],weights=weights))
                        frozen=float(frozen_lookup.loc[(int(row.start),r+1,method,seed,comparison),'raw_mae'])
                        assert np.isclose(actual,frozen,rtol=1e-6,atol=1e-7), (method,row.start,actual,frozen)
                        verified_predictions+=1
                        for axis,lower,upper,lo,hi,mask in masks:
                            count=int(mask.sum());common=mask&statistics['common_positive'];entry=dict(split=row.split,group=int(row.group),start=int(row.start),source_repeat=r+1,comparison=comparison,method=method,seed=seed,axis=axis,distance_lower_bp=lower,distance_upper_bp=upper,layer_lower=lo,layer_upper=hi,blocks=count,unique_pixels=int(weights[mask].sum()),common_positive_blocks=int(common.sum()),status='observed' if count else 'empty')
                            if count:
                                for name in ['raw_mae','raw_mse','target_squared_mean']:entry[name]=float(np.average(statistics[name][mask],weights=weights[mask]))
                                entry['log_mse']=float(np.average(logerror[mask],weights=weights[mask]));entry['conditional_proportion_mse']=float(statistics['conditional_proportion_mse'][common].mean()) if common.any() else None;entry['target_zero_block_fraction']=float(statistics['target_zero'][mask].mean());entry['source_zero_block_fraction']=float((masses[r][mask]==0).mean())
                            rows.append(entry)
            print('Frozen model strata',int(row.start),flush=True)
    strata=pd.DataFrame(rows);strata.to_csv(out/'stratified_per_window.csv',index=False)
    strata_metrics=['raw_mae','raw_mse','target_squared_mean','log_mse','conditional_proportion_mse','target_zero_block_fraction','source_zero_block_fraction']
    groupkeys=['split','comparison','axis','distance_lower_bp','distance_upper_bp','layer_lower','layer_upper','method','group','start','source_repeat']
    sm=strata.groupby(groupkeys,dropna=False)[strata_metrics].mean().reset_index()
    sm.groupby(groupkeys[:8],dropna=False)[strata_metrics].mean().to_csv(out/'stratified_summary.csv')
    effects=paired_effects(sm,['raw_mae','raw_mse','log_mse','conditional_proportion_mse'],groupkeys[:7],['train_distance_prior','bicubic_unprojected','projected_bicubic','uniform'],cfg);effects.to_csv(out/'stratified_paired_intervals.csv',index=False)
    strata.groupby(groupkeys[:8],dropna=False).agg(rows=('blocks','size'),empty_rows=('blocks',lambda x:int((x==0).sum())),total_blocks=('blocks','sum')).to_csv(out/'empty_strata_counts.csv')
    old_noise=pd.read_csv(sixth/'noise/stratified_repeat_agreement.csv');old_noise.to_csv(out/'reused_train_val_noise.csv',index=False)
    ds=pd.read_csv(sixth/'evaluation/known_structure_per_window.csv');ds.groupby(['split','comparison','method','type'])[['oe_100bp_ssim','oe_100bp_pearson','signal_profile_peak_displacement_bp']].mean().to_csv(out/'known_structure_all_methods.csv')
    ds.drop_duplicates('structure_id').groupby('type').size().rename('unique_annotations').to_csv(out/'known_structure_sample_counts.csv')
    aligned=frame[frame.constrained].groupby(['split','start','source_repeat','comparison']).oe_160bp_ssim.agg(lambda x:float(x.max()-x.min()))
    write_json(out/'status.json',dict(main_rows=len(frame),stratified_rows=len(strata),predictions_verified_against_frozen_rows=verified_predictions,all_fixed_strata_included=True,all_empty_cells_preserved=True,all_seeds_preserved=True,all_strata_retrospective=True,trained_new_models=0,raw_interpolation_not_omitted=True,aligned160bp_constrained_max_method_range=float(aligned.max()),known_annotations=int(ds.structure_id.nunique()),count_constraint_is_project_choice_not_teacher_mandate=True,final_formal_choices_modified=False,recommended_new_method=None,reason_not_train='160bp constrained output is algebraically fixed by source40bp totals; sourcezero targetpositive cannot be recovered under exact conservation. Existing shared/conditional networks show tiny cross-repeat differences. No evidence supports another architecture/loss search.',caveat='Conditional proportion error is defined only for common-positive blocks; raw errors retain every block. Effective-pixel strata are paired observed-count diagnostics, not training masks. Stratum MAE/MSE cannot be relabelled as local SSIM.'))
    print('Strong baselines, constraints, full strata and downstream scope audited',flush=True)

if __name__=='__main__':main()

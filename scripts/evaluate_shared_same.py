"""Reuse frozen fifth-round rows; evaluate new arms with precisely its evaluator."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from dlpro.method_review import load_config,sha256,write_json
from dlpro.sr_review import load_data,make_batch,make_model,infer_model
from dlpro.structure_losses import masked_ssim_loss
from dlpro.targeted_review import pearson_all
from evaluate_allocation_review import simple_predictions,metrics,interval


def known_metrics(pred,target,annotations,start,expectation):
    """Annotation-complete local shape/position probes, not a classifier benchmark."""
    n=pred.shape[-1]//10;distance=abs(torch.arange(n)[:,None]-torch.arange(n)[None]);expected=torch.tensor(expectation,dtype=pred.dtype)[distance][None,None]
    pa=torch.log1p(F.avg_pool2d(pred,10)/expected.clamp_min(1e-5));ta=torch.log1p(F.avg_pool2d(target,10)/expected.clamp_min(1e-5))
    rows=[]
    for annotation in annotations.itertuples():
        # Entire annotated interval must be observed. Context is at least seven100bp bins.
        if annotation.start<start or annotation.end>start+pred.shape[-1]*10:continue
        center=int((annotation.center-start)//100)
        lo=min(int((annotation.start-start)//100)-2,center-3);hi=max(int(np.ceil((annotation.end-start)/100))+2,center+4)
        if lo<0 or hi>n:continue
        a,b=pa[:,:,lo:hi,lo:hi],ta[:,:,lo:hi,lo:hi]
        d=abs(torch.arange(hi-lo)[:,None]-torch.arange(hi-lo)[None]);mask=d>=2;upper=torch.triu(mask)
        profile_a=(a[0,0]*mask).sum(1)/mask.sum(1).clamp_min(1);profile_b=(b[0,0]*mask).sum(1)/mask.sum(1).clamp_min(1)
        rows.append(dict(structure_id=annotation.structure_id,type=annotation.type,known_start=int(annotation.start),known_end=int(annotation.end),oe_100bp_ssim=float(1-masked_ssim_loss(a,b,mask)),oe_100bp_pearson=pearson_all(a[0,0][upper].numpy(),b[0,0][upper].numpy()),signal_profile_peak_displacement_bp=abs(int(profile_a.argmax())-int(profile_b.argmax()))*100,roi_bins=hi-lo))
    return rows


def load_models(cfg,params,old=False):
    root=Path(cfg['previous_output'])/'sr_corrected' if old else Path(cfg['output'])/cfg['sr']['result_subdir']
    arms=load_config('configs/method_review_final.json')['sr']['arms'] if old else cfg['sr']['arms']
    models={};hashes={}
    for arm in arms:
        for seed in cfg['sr']['train_seeds']:
            path=root/f'{arm}_seed{seed}.pt';cp=torch.load(path,weights_only=True)
            model=make_model(arm,float(params['scale']),params['profile'],cfg['sr']).eval();model.load_state_dict(cp['state_dict']);models[(arm,seed)]=model
            hashes[path.as_posix()]=sha256(path)
            if not old:
                history=pd.read_csv(root/f'{arm}_seed{seed}_history.csv');assert int(history.loc[history.validation_objective.idxmin(),'step'])==cp['selected_step']
                assert set(cp['training_indices'])==set(params['training_indices'].tolist())
    return models,hashes


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();cfg=load_config(a.config);sr=cfg['sr']
    out=Path(cfg['output'])/'evaluation';out.mkdir(exist_ok=True);assert not (out/'per_window.csv').exists()
    torch.set_num_threads(cfg['threads']);plan,raw,depth=load_data(cfg)
    training=json.loads((Path(cfg['output'])/'runs/train.json').read_text(encoding='utf-8'))
    assert training['status']=='completed' and training['config']==cfg
    for name in ['scripts/train_shared_same.py','src/dlpro/targeted_review.py','src/dlpro/allocation_model.py','src/dlpro/sr_review.py','src/dlpro/refinement.py']:
        assert sha256(name,True)==training['code_sha256'][name],f'Training implementation drift: {name}'
    previous=Path(cfg['previous_output'])/'sr_corrected';params=np.load(previous/'normalization.npz');scale=float(params['scale']);profile=params['profile'];expectations={bp:params[f'expected_{bp}'] for bp in sr['morphology_scales_bp']}
    models,newhash=load_models(cfg,params);oldmodels,oldhash=load_models(cfg,params,True)
    assert json.loads((Path(cfg['output'])/'shared_same/training_status.json').read_text())['completed_models']==len(models)
    values=[]
    with torch.no_grad():
        for index,row in plan[plan.split=='val'].iterrows():
            for r in (0,1):
                high,_=make_batch(raw,depth,[index],[r]);other,_=make_batch(raw,depth,[index],[1-r]);low=F.avg_pool2d(high,4);simple=simple_predictions(low,profile,4)
                for (method,seed),model in models.items():
                    pred=infer_model(model,method,low,scale,sr)[0]
                    for comparison,target in [('same_repeat',high),('cross_repeat',other)]:
                        values.append(dict(split='val',group=int(row.group),start=int(row.start),source_repeat=r+1,method=method,seed=seed,comparison=comparison,constrained=True,**metrics(pred,target,low,simple['projected_bicubic'],scale,expectations,sr)))
            print('shared validation',int(row.start),flush=True)
    oldval=pd.read_csv(previous/'evaluation/validation_per_window.csv');oldval['split']='val'
    validation=pd.concat([oldval,pd.DataFrame(values)],ignore_index=True);validation.to_csv(out/'validation_per_window.csv',index=False)
    means=validation.groupby(['method','comparison'])[['psnr','ssim','oe_100bp_ssim']].mean();means.to_csv(out/'validation_summary.csv')
    alternatives=[]
    for method in sorted(validation.method.unique()):
        eligible=not method.endswith('unprojected')
        for comparison in ('same_repeat','cross_repeat'):
            base=means.loc[('legacy_sparse_mse',comparison)];candidate=means.loc[(method,comparison)]
            eligible &= candidate.psnr>=base.psnr-sr['max_psnr_drop_db'] and candidate.ssim>=base.ssim-sr['max_ssim_drop']
        alternatives.append(dict(method=method,eligible=bool(eligible),cross100=float(means.loc[(method,'cross_repeat'),'oe_100bp_ssim'])))
    choice=max([x for x in alternatives if x['eligible']],key=lambda x:x['cross100'])['method']
    write_json(out/'validation_selection.json',dict(selected_method=choice,alternatives=alternatives,rule='Same as fifth round: constrained, same/cross native safety vs retrained legacy, then maximum cross100bp SSIM; mean seed metrics',completed_utc=datetime.now(timezone.utc).isoformat(),test_data_used=False))
    write_json(out/'test_lock.json',dict(utc=datetime.now(timezone.utc).isoformat(),checkpoints={**oldhash,**newhash},parameter_sha256=sha256(previous/'normalization.npz'),selection_sha256=sha256(out/'validation_selection.json'),historical_rows_sha256=sha256(previous/'evaluation/per_window.csv'),scope='All test positions previously exposed; no independent held-out generalization claim'))
    annotations=pd.read_csv('data/processed/known_windows.csv');rows=[];downstream=[];decomposition=[]
    evaluation=plan[plan.split.isin(['new_test','historical_test'])]
    # Fixed random probes; no selection using gains.
    probes=set(evaluation.groupby('group').head(1).sample(n=2,random_state=cfg['seed']).index)
    with torch.no_grad():
        for index,row in evaluation.iterrows():
            for r in (0,1):
                high,_=make_batch(raw,depth,[index],[r]);other,_=make_batch(raw,depth,[index],[1-r]);low=F.avg_pool2d(high,4);simple=simple_predictions(low,profile,4)
                preds={(name,-1):pred for name,pred in simple.items()}
                for (method,seed),model in {**oldmodels,**models}.items():preds[(method,seed)]=infer_model(model,method,low,scale,sr)[0]
                for (method,seed),pred in preds.items():
                    assert torch.isfinite(pred).all() and pred.min()>=0
                    for comparison,target in [('same_repeat',high),('cross_repeat',other)]:
                        if (method,seed) in models:
                            metric=metrics(pred,target,low,simple['projected_bicubic'],scale,expectations,sr)
                            assert metric['block_mean_max_relative_error']<1e-5 and metric['symmetry_max_relative_error']<1e-5
                            rows.append(dict(split=row.split,group=int(row.group),start=int(row.start),source_repeat=r+1,method=method,seed=seed,comparison=comparison,constrained=True,**metric))
                        for item in known_metrics(pred,target,annotations,int(row.start),expectations[100]):
                            downstream.append(dict(split=row.split,group=int(row.group),start=int(row.start),source_repeat=r+1,method=method,seed=seed,comparison=comparison,**item))
                    upper=torch.triu(torch.ones_like(pred,dtype=torch.bool))
                    midpoint=(high+other)/2
                    lhs=float(((pred-high).square()[upper].mean()+(pred-other).square()[upper].mean())/2)
                    shared=float((pred-midpoint).square()[upper].mean());disagreement=float(((high-other).square()/4)[upper].mean())
                    assert abs(lhs-shared-disagreement)<1e-4*max(lhs,1)
                    decomposition.append(dict(split=row.split,group=int(row.group),start=int(row.start),source_repeat=r+1,method=method,seed=seed,mean_two_target_mse=lhs,error_to_pair_mean=shared,repeat_disagreement_component=disagreement))
                if index in probes and r==0:
                    selected_seed=sr['train_seeds'][0];key=(choice,selected_seed) if (choice,selected_seed) in preds else (choice,-1)
                    np.savez_compressed(out/f'probe_{int(row.start)}.npz',low=low.numpy()[0,0],source=high.numpy()[0,0],target=other.numpy()[0,0],bicubic=simple['projected_bicubic'].numpy()[0,0],prior=simple['train_distance_prior'].numpy()[0,0],shared=preds[('allocation_shared_same',selected_seed)].numpy()[0,0],selected=preds[key].numpy()[0,0])
            print('shared test',int(row.start),flush=True)
    oldframe=pd.read_csv(previous/'evaluation/per_window.csv');frame=pd.concat([oldframe,pd.DataFrame(rows)],ignore_index=True);frame.to_csv(out/'per_window.csv',index=False)
    ds=pd.DataFrame(downstream);ds.to_csv(out/'known_structure_per_window.csv',index=False)
    if len(ds):ds.groupby(['split','comparison','source_repeat','method','type'])[['oe_100bp_ssim','oe_100bp_pearson','signal_profile_peak_displacement_bp']].mean().to_csv(out/'known_structure_summary.csv')
    pd.DataFrame(decomposition).to_csv(out/'repeat_error_decomposition.csv',index=False)
    columns=['psnr','ssim','raw_nrmse','raw_mae','oe_50bp_ssim','oe_100bp_ssim','oe_160bp_ssim']
    frame.groupby(['split','comparison','source_repeat','method','seed'])[columns].mean().to_csv(out/'directional_summary.csv')
    frame.groupby(['split','comparison','method'])[columns].mean().to_csv(out/'summary.csv')
    mean=frame.groupby(['split','comparison','method','group','start','source_repeat'])[columns].mean().reset_index();effects=[]
    for (split,comparison),part in mean.groupby(['split','comparison']):
        for method in sr['arms']:
            for ref in ['allocation_single_same','allocation_expanded_cross','allocation_expanded_mixed','context_expanded_shape','legacy_sparse_mse','train_distance_prior','projected_bicubic','uniform']:
                if ref==method:continue
                pair=part[part.method==method].merge(part[part.method==ref],on=['group','start','source_repeat'],suffixes=('_a','_b'),validate='one_to_one')
                for metric in columns:effects.append(dict(split=split,comparison=comparison,method=method,reference=ref,metric=metric,**interval((pair[metric+'_a']-pair[metric+'_b']).to_numpy(),pair.group,cfg)))
    pd.DataFrame(effects).to_csv(out/'paired_intervals.csv',index=False)
    # Seed direction stability is reported, never choose the higher test seed.
    frame[frame.method.isin(sr['arms'])].groupby(['split','comparison','method','seed'])[columns].mean().to_csv(out/'seed_stability.csv')
    write_json(out/'checks.json',dict(new_models=len(models),reused_models=len(oldmodels),new_rows=len(rows),total_rows=len(frame),known_structure_rows=len(ds),known_unique_structures=int(ds.structure_id.nunique()) if len(ds) else 0,new_max_count_error=float(frame[frame.method.isin(sr['arms'])].block_mean_max_relative_error.max()),new_max_symmetry_error=float(frame[frame.method.isin(sr['arms'])].symmetry_max_relative_error.max()),all_nonnegative=True,downstream_definition='Complete annotation local100bp OE shape and profile-peak displacement; not full-context classifier accuracy',selection=choice,seed_aggregation='Mean metrics, not an ensemble',historical_exploration=True))

if __name__=='__main__':main()

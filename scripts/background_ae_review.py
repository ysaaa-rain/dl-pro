"""Supported AE reconstruction, with training-only nuisance calibration and LOTO."""
import argparse
import copy
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits
from dlpro.coverage_support import coarse_support
from dlpro.experiment_core import Autoencoder, LENGTH, configure, hits, centered_indices
from dlpro.method_review import load_config, nms, spatial_cv_masks, sha256, write_json
from dlpro.novelty import empirical_rank, grouped_auc, shape_vectors
from dlpro.refinement import morphology_features
from dlpro.ridge_features import directional_features
from dlpro.targeted_review import shape_input, masked_errors, NuisanceResidual


def expected_profile(raw, support, ids):
    n=raw.shape[-1];d=abs(np.arange(n)[:,None]-np.arange(n)[None]);expected=[]
    for channel in range(raw.shape[1]):
        mask=support[ids,channel,:,None]&support[ids,channel,None,:]
        image=raw[ids,channel]
        expected.append([max(float(image[:,d==k][mask[:,d==k]].mean()) if mask[:,d==k].any() else 0,1e-8) for k in range(n)])
    return np.asarray(expected)


def train_one(x,mask,train,val,seed,ac,directory,threads):
    configure(seed);torch.set_num_threads(threads);model=Autoencoder()
    optimizer=torch.optim.Adam(model.parameters(),lr=ac['learning_rate']);rng=np.random.default_rng(seed)
    tx,tm=torch.from_numpy(x),torch.from_numpy(mask);history=[];best=float('inf');begin=time.monotonic()
    completed=directory/f'seed{seed}.pt'
    if completed.exists():
        cp=torch.load(completed,weights_only=True)
        assert cp['train_indices']==train.tolist() and cp['validation_indices']==val.tolist()
        state=cp['state_dict'];selected=cp['selected_epoch']
        history=pd.read_csv(directory/f'seed{seed}_history.csv').to_dict('records')
        best=history[selected-1]['validation_mse']
    for epoch in ([] if completed.exists() else range(1,ac['epochs']+1)):
        model.train();values=[]
        for ids in np.array_split(rng.permutation(train),int(np.ceil(len(train)/ac['batch_size']))):
            optimizer.zero_grad();loss=masked_errors(model(tx[ids]),tx[ids],tm[ids]).mean()
            assert torch.isfinite(loss);loss.backward();optimizer.step();values.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():value=float(masked_errors(model(tx[val]),tx[val],tm[val]).mean())
        history.append(dict(epoch=epoch,train_mse=float(np.mean(values)),validation_mse=value))
        if value<best:best=value;selected=epoch;state=copy.deepcopy(model.state_dict())
    model.load_state_dict(state);model.eval();errors=[];embeddings=[]
    with torch.no_grad():
        for ids in np.array_split(np.arange(len(x)),int(np.ceil(len(x)/ac['batch_size']))):
            errors.extend(masked_errors(model(tx[ids]),tx[ids],tm[ids]).numpy())
            embeddings.extend(model.encoder(tx[ids]).numpy())
    if not completed.exists():
        pd.DataFrame(history).to_csv(directory/f'seed{seed}_history.csv',index=False)
        torch.save(dict(state_dict=state,selected_epoch=selected,seed=seed,train_indices=train.tolist(),validation_indices=val.tolist()),completed)
    diag=dict(seed=seed,selected_epoch=selected,train_selected=history[selected-1]['train_mse'],validation_selected=best,elapsed_seconds=time.monotonic()-begin)
    if (directory/'selection_lock.json').exists():
        diag=next(v for v in json.loads((directory/'selection_lock.json').read_text())['seed_diagnostics'] if v['seed']==seed)
    return np.asarray(errors),np.asarray(embeddings),diag


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();cfg=load_config(a.config)
    root=Path(cfg['output']);out=root/cfg['discovery']['result_subdir'];out.mkdir(exist_ok=True)
    assert not (out/'point_scores.csv').exists()
    dc,ac=cfg['discovery'],cfg['ae'];threadpool_limits(2)
    known=pd.read_csv(root/'data/known_eligibility.csv');scan=pd.read_csv(root/'data/scan_eligibility.csv');nk=len(known)
    centers=np.r_[known.center,scan.center];folds=np.r_[known.new_fold,scan.new_fold];groups=np.r_[known.spatial_block,scan.spatial_block]
    eligible=np.r_[known.eligible,scan.eligible].astype(bool)&(folds>=0);is_scan=np.arange(len(centers))>=nk
    types=np.r_[known.type,np.full(len(scan),'background')]
    raw={r:np.concatenate([np.load(f'data/processed/known_rep{r}.npz')['raw'],np.load(f'data/processed/scan_rep{r}.npz')['raw']]) for r in (1,2)}
    support={r:coarse_support(np.load(f'data/processed/coverage_rep{r}.npz')['coverage'],centers)>=dc['support_fraction'] for r in (1,2)}
    previous=Path(cfg['previous_output'])/'novelty_corrected'
    fits=json.loads((previous/'fits.json').read_text(encoding='utf-8'))
    points=[];recalls=[];selected=[];diagnostics=[];confounds=[];newfits=[];locks=[]
    # All backgrounds and references exactly match the corrected fifth-round masks.
    for fit in fits:
        hidden,fold=fit['hidden_type'],fit['fold'];directory=out/f'{hidden}_fold{fold}';directory.mkdir(exist_ok=True)
        train=np.asarray(fit['background_training_indices'],int);refs=np.asarray(fit['reference_indices'],int)
        val=np.asarray(fit['calibration_background_indices'],int);valrefs=np.asarray(fit['calibration_reference_indices'],int)
        training,validation,test=spatial_cv_masks(eligible,folds,fold,dc['folds'])
        assert training[train].all() and training[refs].all() and validation[val].all()
        assert (types[refs]!=hidden).all() and (types[valrefs]!=hidden).all()
        assert (folds[np.r_[train,refs,val,valrefs]]>=0).all()
        expected=expected_profile(raw[1],support[1],train)
        _,nuisance1=shape_vectors(raw[1],support[1],expected,dc['pooled_bins'])
        _,nuisance2=shape_vectors(raw[2],support[2],expected,dc['pooled_bins']);nuisance={1:nuisance1,2:nuisance2}
        if (directory/'transform.npz').exists():np.testing.assert_array_equal(np.load(directory/'transform.npz')['expected'],expected)
        else:np.savez_compressed(directory/'transform.npz',expected=expected,train_indices=train,val_indices=val)
        # The exact incumbent scoring is refitted using its old, allowed-label choice.
        needed=np.unique(np.r_[train,refs,val,valrefs,np.flatnonzero(test)])
        morph={}
        for r in (1,2):
            part=morphology_features(raw[r][needed],expected,support[r][needed])
            morph[r]=np.zeros((len(centers),part.shape[1]),dtype=part.dtype);morph[r][needed]=part
        z={r:np.empty_like(morph[r]) for r in (1,2)}
        for col in range(morph[1].shape[1]):
            model=HuberRegressor(max_iter=1000).fit(nuisance[1][train,:1],np.log(morph[1][train,col].clip(1e-6)))
            residual=np.log(morph[1][train,col].clip(1e-6))-model.predict(nuisance[1][train,:1]);mu=np.median(residual);sd=max(1.4826*np.median(abs(residual-mu)),dc['scale_floor'])
            for r in (1,2):z[r][:,col]=(np.log(morph[r][:,col].clip(1e-6))-model.predict(nuisance[r][:,:1])-mu)/sd
        def aggregate(v):return dict(maximum=v.max(1),mean=v.mean(1),positive_mean=np.maximum(v,0).mean(1))[fit['morphology_choice']]
        base={r:empirical_rank(aggregate(z[r]),aggregate(z[1])[val]) for r in (1,2)}
        direction={}
        for r in (1,2):
            part=directional_features(raw[r][needed],expected,support[r][needed])
            direction[r]=np.zeros((len(centers),part.shape[1]),dtype=part.dtype);direction[r][needed]=part
        directional_cal=NuisanceResidual(ac['calibration_ridge']).fit(nuisance[1][train],np.log(direction[1][train].clip(1e-6)))
        direction_score={r:np.maximum(directional_cal.predict(nuisance[r],np.log(direction[r].clip(1e-6))),0).mean(1) for r in (1,2)}
        direction_rank={r:empirical_rank(direction_score[r],direction_score[1][val]) for r in (1,2)}
        # Refit scalar only from allowed training backgrounds, never the whole genome.
        _,_,scale=shape_input(raw[1][train,ac['channel']],support[1][train,ac['channel']],expected[ac['channel']],quantile=ac['quantile'])
        x1,m1,_=shape_input(raw[1][:,ac['channel']],support[1][:,ac['channel']],expected[ac['channel']],scale)
        x2,m2,_=shape_input(raw[2][:,ac['channel']],support[2][:,ac['channel']],expected[ac['channel']],scale)
        residual_ranks={1:[],2:[]};latent_ranks={1:[],2:[]};seed_diagnostics=[]
        for seed in ac['seeds']:
            errors1,embedding1,diag=train_one(x1,m1,train,val,seed,ac,directory,cfg['threads'])
            model=Autoencoder().eval();model.load_state_dict(torch.load(directory/f'seed{seed}.pt',weights_only=True)['state_dict'])
            errors2=[];embedding2=[]
            with torch.no_grad():
                for ids in np.array_split(np.arange(len(x2)),int(np.ceil(len(x2)/ac['batch_size']))):
                    tx=torch.from_numpy(x2[ids]);tm=torch.from_numpy(m2[ids]);errors2.extend(masked_errors(model(tx),tx,tm).numpy());embedding2.extend(model.encoder(tx).numpy())
            errors={1:errors1,2:np.asarray(errors2)};emb={1:embedding1,2:np.asarray(embedding2)}
            calibrator=NuisanceResidual(ac['calibration_ridge']).fit(nuisance[1][train],np.log(errors1[train].clip(1e-12)))
            score={r:calibrator.predict(nuisance[r],np.log(errors[r].clip(1e-12))) for r in (1,2)}
            scaler=StandardScaler().fit(emb[1][train]);vectors={r:scaler.transform(emb[r]) for r in (1,2)}
            centroids=np.stack([np.median(vectors[1][refs[groups[refs]==g]],axis=0) for g in np.unique(groups[refs])])
            distance={r:cdist(vectors[r],centroids).min(1) for r in (1,2)}
            for r in (1,2):
                residual_ranks[r].append(empirical_rank(score[r],score[1][val]));latent_ranks[r].append(empirical_rank(distance[r],distance[1][val]))
            seed_diagnostics.append(diag)
            for r in (1,2):
                for i in np.flatnonzero(test):
                    diagnostics.append(dict(hidden_type=hidden,fold=fold,seed=seed,repeat=r,item_id=known.structure_id.iloc[i] if i<nk else f'scan_{i-nk}',actual_type=types[i],raw_mse=float(errors[r][i]),calibrated_residual=float(score[r][i]),residual_rank=float(residual_ranks[r][-1][i])))
        ae={r:np.mean(residual_ranks[r],axis=0) for r in (1,2)}
        latent={r:np.mean(latent_ranks[r],axis=0) for r in (1,2)}
        joint={r:(1-ac['joint_weight'])*ae[r]+ac['joint_weight']*direction_rank[r] for r in (1,2)}
        label=np.r_[np.ones(len(valrefs)),np.zeros(len(val))];indices=np.r_[valrefs,val]
        val_auc={name:float(roc_auc_score(label,score[indices])) for name,score in [('ae',ae[1]),('morphology',base[1]),('joint',joint[1])]}
        correlation=float(spearmanr(ae[1][val],direction_rank[1][val]).statistic)
        joint_allowed=bool(val_auc['joint']>=max(val_auc['ae'],val_auc['morphology'])+ac['joint_min_validation_auc_gain'] and np.isfinite(correlation) and abs(correlation)<ac['joint_max_validation_spearman'])
        # Rejected fusion is not silently used for production candidates.
        methods={r:dict(morphology_baseline=base[r],ae_residual=ae[r],ae_direction_joint=joint[r] if joint_allowed else ae[r],ae_reference_diagnostic=np.minimum(ae[r],latent[r])) for r in (1,2)}
        checkpoint_hashes={f'seed{s}':sha256(directory/f'seed{s}.pt') for s in ac['seeds']}
        freeze=dict(hidden_type=hidden,fold=fold,validation_auc=val_auc,validation_signal_spearman=correlation,joint_allowed=joint_allowed,joint_fallback='ae_residual',checkpoints=checkpoint_hashes,scale=scale,seed_diagnostics=seed_diagnostics,test_labels_used=False)
        if (directory/'selection_lock.json').exists():
            original=json.loads((directory/'selection_lock.json').read_text())
            assert original==freeze, 'Resume must reproduce the original validation selection exactly'
        else:write_json(directory/'selection_lock.json',freeze)
        locks.append(freeze)
        nf=dict(fit);nf.update(freeze);newfits.append(nf)
        # The lock is written BEFORE scoring test labels/recall for this fit.
        test_points=np.flatnonzero(test&(~is_scan|np.array([not hits(int(c),dc['max_context_bp'],known,buffer=dc['known_buffer_bp']).any() for c in centers])))
        test_scan=np.flatnonzero(test&is_scan);test_known=np.flatnonzero(test&~is_scan)
        arrays={f'{name}_rep{r}':values for r,m in methods.items() for name,values in m.items()}
        if (directory/'all_scores.npz').exists():
            previous_arrays=np.load(directory/'all_scores.npz')
            for key,value in arrays.items():np.testing.assert_array_equal(previous_arrays[key][needed],value[needed])
        else:np.savez_compressed(directory/'all_scores.npz',**arrays)
        for r in (1,2):
            for name,score in methods[r].items():
                threshold=float(np.quantile(score[val],1-dc['false_positive_rate']))
                for i in test_points:
                    points.append(dict(hidden_type=hidden,fold=fold,repeat=r,method=name,item_id=known.structure_id.iloc[i] if i<nk else f'scan_{i-nk}',actual_type=types[i],group=int(groups[i]),score=float(score[i]),flagged=bool(score[i]>threshold),calibration_threshold=threshold))
                for field,values in zip(['log_density','zero_fraction','support'],nuisance[r][test_scan,:3].T):
                    value=float(spearmanr(score[test_scan],values).statistic)
                    confounds.append(dict(hidden_type=hidden,fold=fold,repeat=r,method=name,nuisance=field,spearman=value if np.isfinite(value) else None))
                if name=='ae_reference_diagnostic':continue
                for global_budget in dc['global_budgets']:
                    budget=int(np.ceil(global_budget*len(test_scan)/len(scan)))
                    picked=test_scan[nms(score[test_scan],centers[test_scan],budget,dc['nms_bp'],LENGTH)]
                    delta=abs(centers[picked,None]-centers[test_known][None]);hit=(np.minimum(delta,LENGTH-delta)<=dc['hit_radius_bp']).any(0)
                    random_rng=np.random.default_rng(cfg['seed']+fold)
                    random_hits=[]
                    for _ in range(ac['random_trials']):
                        rr=test_scan[nms(random_rng.random(len(test_scan)),centers[test_scan],budget,dc['nms_bp'],LENGTH)]
                        d=abs(centers[rr,None]-centers[test_known][None]);random_hits.append((np.minimum(d,LENGTH-d)<=dc['hit_radius_bp']).any(0))
                    random_probability=np.mean(random_hits,axis=0)
                    for i,h,prob in zip(test_known,hit,random_probability):
                        recalls.append(dict(hidden_type=hidden,fold=fold,repeat=r,method=name,global_equivalent_budget=global_budget,actual_fold_budget=budget,structure_id=known.structure_id.iloc[i],actual_type=types[i],group=int(groups[i]),recalled=bool(h),random_hit_probability=float(prob)))
                    for rank,i in enumerate(picked,1):
                        selected.append(dict(hidden_type=hidden,fold=fold,repeat=r,method=name,global_equivalent_budget=global_budget,rank=rank,scan_index=int(i-nk),center=int(centers[i]),score=float(score[i])))
        print('AE complete',hidden,fold,'background',len(train),'val',val_auc,'fusion',joint_allowed,flush=True)
    frame=pd.DataFrame(points);frame.to_csv(out/'point_scores.csv',index=False)
    pd.DataFrame(recalls).to_csv(out/'budget_recall.csv',index=False);pd.DataFrame(selected).to_csv(out/'selected_scan_windows.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(out/'reconstruction_per_window.csv',index=False);pd.DataFrame(confounds).to_csv(out/'nuisance_correlations.csv',index=False)
    write_json(out/'fits.json',newfits);write_json(out/'selection_locks.json',locks)
    metrics=[]
    for (hidden,r,name),part in frame.groupby(['hidden_type','repeat','method']):
        for negative in ('allowed_known','background'):
            sub=part[(part.actual_type==hidden)|((part.actual_type=='background') if negative=='background' else ((part.actual_type!=hidden)&(part.actual_type!='background')))]
            y=sub.actual_type.eq(hidden).to_numpy();estimate=grouped_auc(y,sub.score.to_numpy(),sub.group.to_numpy(),cfg['seed'],cfg['bootstrap_draws'])
            metrics.append(dict(hidden_type=hidden,repeat=r,method=name,negative=negative,n_hidden=int(y.sum()),n_negative=int((~y).sum()),average_precision=float(average_precision_score(y,sub.score)),hidden_detection=float(sub.loc[y,'flagged'].mean()),false_positive_rate=float(sub.loc[~y,'flagged'].mean()),**estimate))
    pd.DataFrame(metrics).to_csv(out/'novelty_summary.csv',index=False)
    write_json(out/'status.json',dict(fits=len(fits),models=len(fits)*len(ac['seeds']),fusion_allowed_fits=sum(x['joint_allowed'] for x in locks),background_channel_bp=dc['input_scales_bp'][ac['channel']],same_partition_as_fifth_round=True,retrospective=True,repeat2_not_used_for_fitting=True,score='Mean of two fixed-seed residual ECDF ranks; joint falls back when validation complementarity gate fails',reference_diagnostic='Latent reference novelty is separate diagnostic, never candidate selection',all_known_available=int(eligible[:nk].sum()),scan_available=int(eligible[nk:].sum())))

if __name__=='__main__':main()

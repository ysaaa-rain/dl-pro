"""Purged spatial outer-fold evaluation of fixed-budget morphology and background AE."""
from pathlib import Path
import copy
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from sklearn.linear_model import HuberRegressor
from sklearn.isotonic import IsotonicRegression
from dlpro.experiment_core import DensityInvariantTransform,Autoencoder,configure,save_json,LENGTH,centered_indices
from dlpro.refinement import morphology_features

out=Path('results/discovery_spatial_cv');out.mkdir(exist_ok=True)
known=pd.read_csv('data/processed/known_windows.csv');scan=pd.read_csv('data/processed/scan_windows.csv');table=pd.read_csv('results/classification_cv/out_of_fold_predictions.csv').set_index('structure_id');folds=table.loc[known.structure_id,'fold'].to_numpy();raw=np.load('data/processed/scan_rep1.npz')['raw'];centers=scan.center.to_numpy();delta=abs(centers[:,None]-known.center.to_numpy()[None]);coverage=np.minimum(delta,LENGTH-delta)<=6400;starts=np.asarray([centered_indices(v)[1]*200 for v in centers]);ends=starts+25600

def select(score,budget):
    ids=[];available=np.ones(len(scan),bool)
    for i in np.argsort(-score,kind='stable'):
        if not available[i]:continue
        ids.append(i);d=abs(centers-centers[i]);available&=np.minimum(d,LENGTH-d)>=12800
        if len(ids)==budget:return np.asarray(ids)
    raise ValueError('Budget exceeds available nonoverlap windows')

rng=np.random.default_rng(314);random_cover={}
for budget in [64,128]:random_cover[budget]=np.asarray([coverage[select(rng.random(len(scan)),budget)].any(0) for _ in range(500)])
random_known=[]
for budget,v in random_cover.items():
    for i,r in known.iterrows():random_known.append(dict(structure_id=r.structure_id,budget=budget,random_hit_probability=float(v[:,i].mean())))
pd.DataFrame(random_known).to_csv(out/'random_locus_probabilities.csv',index=False)
records=[];candidate_rows=[];fold_metrics=[];protocol=[]
for fold in range(5):
    configure(42);test=folds==fold;val=folds==(fold+1)%5;held=known[test|val];overlap=(starts[:,None]<held.window_end.to_numpy()[None])&(ends[:,None]>held.window_start.to_numpy()[None]);eligible=scan.training_background.to_numpy()&~overlap.any(1)&(starts>=0)&(ends<=LENGTH)
    blocks=scan.loc[eligible,'background_block'].unique();order=np.random.default_rng(42+fold).permutation(blocks);cut=max(1,int(.7*len(order)));train=eligible&scan.background_block.isin(order[:cut]).to_numpy();bg_val=eligible&scan.background_block.isin(order[cut:]).to_numpy()
    assert not overlap[train|bg_val].any() and train.sum()>=10 and bg_val.sum()>=5
    directory=out/f'fold{fold}';directory.mkdir(exist_ok=True);tf=DensityInvariantTransform().fit(raw[train]);tf.save(directory/'transform.npz');features=morphology_features(raw,tf.expected);density=np.log(np.maximum(raw[:,0].mean((1,2)),1e-12));z=np.empty_like(features);calibration=[]
    for col in range(features.shape[1]):
        response=np.log(np.maximum(features[:,col],1e-6));fit=HuberRegressor().fit(density[train,None],response[train]);residual=response[train]-fit.predict(density[train,None]);location=float(np.median(residual));scale=max(float(1.4826*np.median(abs(residual-location))),.05);z[:,col]=(response-fit.predict(density[:,None])-location)/scale;calibration.append(dict(feature=col,slope=float(fit.coef_[0]),intercept=float(fit.intercept_),location=location,scale=scale))
    alternatives={'maximum':z.max(1),'mean':z.mean(1),'positive_mean':np.maximum(z,0).mean(1)};validation={m:float(coverage[select(v,64)].any(0)[val].mean()) for m,v in alternatives.items()};choice=max(alternatives,key=lambda m:validation[m]);morph=alternatives[choice]
    tx=torch.from_numpy(tf.transform(raw)[:,:1]);model=Autoencoder();optimizer=torch.optim.Adam(model.parameters(),lr=.001);rng=np.random.default_rng(42);best=float('inf');state=None;history=[]
    for epoch in range(30):
        model.train();losses=[]
        for ids in np.array_split(rng.permutation(np.flatnonzero(train)),int(np.ceil(train.sum()/16))):
            optimizer.zero_grad();loss=F.mse_loss(model(tx[ids]),tx[ids]);loss.backward();optimizer.step();losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():vl=float(F.mse_loss(model(tx[bg_val]),tx[bg_val]))
        history.append(dict(epoch=epoch+1,train_mse=float(np.mean(losses)),val_mse=vl))
        if vl<best:best=vl;state=copy.deepcopy(model.state_dict());selected_epoch=epoch+1
    model.load_state_dict(state);model.eval()
    with torch.no_grad():reconstructed=np.concatenate([model(batch).numpy() for batch in tx.split(64)])
    error=np.log(np.maximum(((reconstructed-tx.numpy())**2).mean((1,2,3)),1e-12));null=IsotonicRegression(increasing=False,out_of_bounds='clip').fit(density[train],error[train]);edges=np.quantile(density[train],[.2,.4,.6,.8]);bucket=np.searchsorted(edges,density);resid=error[train]-null.predict(density[train]);location=[];spread=[]
    for b in range(5):
        v=resid[bucket[train]==b];mu=float(np.median(v));location.append(mu);spread.append(max(float(1.4826*np.median(abs(v-mu))),.001))
    ae=(error-null.predict(density)-np.asarray(location)[bucket])/np.asarray(spread)[bucket];assert np.isfinite(ae).all() and np.isfinite(morph).all()
    scores={'morphology':morph,'noise_adjusted_ae':ae,'density':density};score_table=scan[['center']].copy()
    for method,score in scores.items():
        score_table[method]=score
        for budget in [64,128]:
            ids=select(score,budget);hit=coverage[ids].any(0)
            for rank,i in enumerate(ids):candidate_rows.append(dict(fold=fold,method=method,budget=budget,rank=rank+1,scan_index=int(i),center=int(centers[i]),score=float(score[i])))
            for i in np.flatnonzero(test):records.append(dict(structure_id=known.structure_id.iloc[i],type=known.type.iloc[i],group=int(known.group.iloc[i]),fold=fold,method=method,budget=budget,recalled=bool(hit[i]),random_hit_probability=float(random_cover[budget][:,i].mean())))
            entry=dict(fold=fold,method=method,budget=budget,n_test=int(test.sum()),n_background_train=int(train.sum()),n_background_val=int(bg_val.sum()),val_center_recall=float(hit[val].mean()),test_center_recall=float(hit[test].mean()),random_test_mean=float(random_cover[budget][:,test].mean()));fold_metrics.append(entry);print(entry,flush=True)
    score_table.to_csv(directory/'all_scores.csv',index=False);pd.DataFrame(history).to_csv(directory/'ae_history.csv',index=False);torch.save(dict(state_dict=state,selected_epoch=selected_epoch),directory/'ae.pt');np.savez_compressed(directory/'features.npz',features=features,standardized=z,background_train=train,background_val=bg_val,scan_window_start=starts,scan_window_end=ends)
    np.savez(directory/'ae_null.npz',density_thresholds=null.X_thresholds_,error_thresholds=null.y_thresholds_,edges=edges,location=location,spread=spread)
    protocol.append(dict(fold=fold,n_background_train=int(train.sum()),n_background_val=int(bg_val.sum()),held_input_overlap_purged=True,train_blocks=order[:cut].astype(int).tolist(),val_blocks=order[cut:].astype(int).tolist(),morphology_choice=choice,validation_choices=validation,density_calibration=calibration,ae_selected_epoch=selected_epoch))
frame=pd.DataFrame(records);frame.to_csv(out/'out_of_fold_recall.csv',index=False);pd.DataFrame(candidate_rows).to_csv(out/'candidates.csv',index=False);pd.DataFrame(fold_metrics).to_csv(out/'fold_metrics.csv',index=False)
summary=[];rng=np.random.default_rng(541)
for (method,budget),g in frame.groupby(['method','budget']):
    assert len(g)==344 and g.structure_id.nunique()==344;groups=g.group.unique();draw=[]
    for _ in range(2000):
        chosen=rng.choice(groups,len(groups),replace=True);idx=np.concatenate([np.flatnonzero(g.group.to_numpy()==v) for v in chosen]);draw.append((g.recalled.to_numpy()[idx].astype(float)-g.random_hit_probability.to_numpy()[idx]).mean())
    entry=dict(method=method,budget=int(budget),n_loci=len(g),center_recall=float(g.recalled.mean()),random_mean=float(g.random_hit_probability.mean()),delta_group_bootstrap_95=np.quantile(draw,[.025,.975]).tolist(),n_groups=len(groups));summary.append(entry)
save_json(out/'summary.json',summary);save_json(out/'protocol.json',dict(seed=42,outer_folds=5,budgets=[64,128],nms_bp=12800,random_trials=500,fold_protocols=protocol,scope='Known-structure center recovery proxy; not biological novelty detection',limitations=['Partly supervised choice of morphology aggregation on validation known structures','Previously examined outer test labels; retrospective cross-validation','Background excludes full25.6kb held-out known inputs but genomic scan ranking is transductive','Same-study spatial groups, not independent source or biological replicate bootstrap']))
print(summary,flush=True)

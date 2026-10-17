"""Two audit-directed mechanisms, frozen outer groups and strictly allowed fitting rows."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from dlpro.method_review import load_config,write_json,sha256
from dlpro.experiment_core import LABELS,ShapeTransform
from dlpro.classification_models import FineScaleClassifier
from dlpro.shape_features import summarize_shapes
from dlpro.classification_review import load_inputs,metrics,choose_bias,calibrate_bias,orientation_features,paired_metric_intervals,check_spatial_partition

def probability(models,images):
    with torch.no_grad():return np.mean([np.concatenate([m(b).softmax(1).numpy() for b in torch.from_numpy(images).split(64)]) for m in models],axis=0)

def fit_logistic(x,y,C,cc):
    scaler=StandardScaler().fit(x);model=LogisticRegression(C=C,class_weight='balanced',max_iter=cc['logistic_max_iter'],random_state=42).fit(scaler.transform(x),y)
    assert int(model.n_iter_.max())<cc['logistic_max_iter'];return scaler,model

def features(raw,support,training):
    transform=ShapeTransform().fit(raw[training]);geometric=summarize_shapes(transform.transform(raw));direction=orientation_features(raw,transform.expected,support)
    return transform,{'geometric_control':geometric,'geometric_directional':np.c_[geometric,direction]}

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();cfg=load_config(a.config);root=Path(cfg['output']);out=root/'experiment';out.mkdir(exist_ok=True);assert not (out/'status.json').exists();assert (root/'audit/status.json').exists();cc=cfg['classifier'];assert cc['methods_frozen_after_audit'];torch.set_num_threads(cfg['threads'])
    frame,folds,raw,support=load_inputs();n=len(frame);k=len(LABELS);y=frame.type.map(dict(zip(LABELS,range(k)))).to_numpy();yy=np.tile(y,2);valid=support>=cc['support_fraction'];old=pd.read_csv(Path(cc['source'])/'out_of_fold_predictions.csv');old=old[old.recipe==cc['baseline']].set_index('structure_id');predictions=[];inner_scores=[];partitions=[];frozen=[]
    for fold in range(cc['outer_folds']):
        directory=out/f'fold{fold}';directory.mkdir(exist_ok=True);test=folds==fold;val=folds==(fold+1)%cc['outer_folds'];train=~(test|val);train_views=np.tile(train,2);test_ids=np.flatnonzero(test);val_ids=np.flatnonzero(val)
        roles=np.where(test,'outer_test',np.where(val,'inner_validation','training'));check_spatial_partition(frame.window_start.to_numpy(),frame.window_end.to_numpy(),roles)
        for i,row in frame.iterrows():partitions.append(dict(outer_fold=fold,structure_id=row.structure_id,group=int(row.group),role=roles[i],existing_fold=int(folds[i])))
        transform=ShapeTransform.load(Path(cc['source'])/f'transform_fold{fold}.npz');refit=ShapeTransform().fit(raw[train_views]);np.testing.assert_allclose(transform.expected,refit.expected,rtol=1e-6);np.testing.assert_allclose(transform.scale,refit.scale,rtol=1e-6)
        write_json(directory/'normalization_refit_check.json',dict(max_expected_absolute_difference=float(abs(transform.expected-refit.expected).max()),max_scale_absolute_difference=float(abs(transform.scale-refit.scale).max()),relative_tolerance=1e-6,reason='Float32 quantile reduction can differ at a few machine epsilon across NumPy/runtime versions; frozen transform is always used for CNN inference'))
        models=[];hashes={}
        for seed in cc['seeds']:
            path=Path(cc['source'])/f'fine_consistent_seed{seed}_fold{fold}.pt';cp=torch.load(path,weights_only=True);model=FineScaleClassifier();model.load_state_dict(cp['state_dict']);model.eval();models.append(model);hashes[path.as_posix()]=sha256(path)
        cnn_view=probability(models,transform.transform(raw));cnn=(cnn_view[:n]+cnn_view[n:])/2;expected=old.loc[frame.loc[test,'structure_id'],[f'p_{label}' for label in LABELS]].to_numpy();np.testing.assert_allclose(cnn[test],expected,rtol=1e-6,atol=1e-7)
        bias,trials=choose_bias(cnn[val],y[val],cc['bias_grid']);calibrated=calibrate_bias(cnn,bias);selected={'frozen_cnn':cnn,'symmetric_bias':calibrated};calibration=dict(bias=bias.tolist(),trials=trials,validation_indices=val_ids.tolist(),test_labels_used=False);write_json(directory/'calibration_lock.json',calibration)
        # Both feature arms use identical three inner group folds; no outer-val/test fitted transforms.
        inner_predictions={name:{C:np.full((n,k),np.nan) for C in cc['logistic_C']} for name in ['geometric_control','geometric_directional']}
        for inner in sorted(np.unique(folds[train])):
            iv=train&(folds==inner);it=train&~iv;training=np.tile(it,2);held=np.flatnonzero(iv)
            itf,all_features=features(raw,valid,training);itf.save(directory/f'inner{inner}_transform.npz')
            write_json(directory/f'inner{inner}_roles.json',dict(train_indices=np.flatnonzero(it).tolist(),validation_indices=held.tolist(),outer_excluded_indices=np.flatnonzero(test|val).tolist()))
            for name,values in all_features.items():
                for C in cc['logistic_C']:
                    scaler,model=fit_logistic(values[training],yy[training],C,cc);p=model.predict_proba(scaler.transform(values[np.r_[held,held+n]]));p=(p[:len(held)]+p[len(held):])/2;inner_predictions[name][C][held]=p
                    inner_scores.append(dict(outer_fold=fold,inner_fold=int(inner),method=name,C=C,macro_f1=float(metrics(y[iv],p.argmax(1),k)['macro_f1']),training_loci=int(it.sum()),validation_loci=int(iv.sum())))
            print('outer',fold,'inner',inner,'features and paired arms complete',flush=True)
        final_transform,all_features=features(raw,valid,train_views);final_transform.save(directory/'final_feature_transform.npz');choices={}
        for name,values in all_features.items():
            grid=[]
            for C in cc['logistic_C']:
                sub=[r['macro_f1'] for r in inner_scores if r['outer_fold']==fold and r['method']==name and r['C']==C];grid.append((float(np.mean(sub)),-C,C))
            C=max(grid)[2];scaler,model=fit_logistic(values[train_views],yy[train_views],C,cc);joblib.dump(dict(scaler=scaler,model=model,feature_recipe=name,train_indices=np.flatnonzero(train).tolist(),C=C),directory/f'{name}.joblib',compress=3)
            p=model.predict_proba(scaler.transform(values));selected[name]=(p[:n]+p[n:])/2;choices[name]=dict(C=C,inner_grid=[dict(C=item[2],mean_macro_f1=item[0]) for item in grid])
        lock=dict(outer_fold=fold,training_indices=np.flatnonzero(train).tolist(),validation_indices=val_ids.tolist(),test_indices=test_ids.tolist(),checkpoints=hashes,calibration_bias=bias.tolist(),feature_choices=choices,all_choices_without_outer_test_labels=True,architecture_search=False)
        write_json(directory/'selection_lock.json',lock);frozen.append(lock)
        for name,prob in selected.items():
            for i in test_ids:
                row=dict(structure_id=frame.structure_id.iloc[i],true_type=frame.type.iloc[i],group=int(frame.group.iloc[i]),fold=fold,method=name,predicted_type=LABELS[prob[i].argmax()]);row.update({f'p_{label}':float(prob[i,c]) for c,label in enumerate(LABELS)});predictions.append(row)
        np.savez_compressed(directory/'frozen_cnn_probabilities.npz',per_view=cnn_view,paired=cnn,train_indices=np.flatnonzero(train),validation_indices=val_ids,test_indices=test_ids)
        pd.DataFrame([r for r in predictions if r['fold']==fold]).to_csv(directory/'outer_predictions.csv',index=False)
        print('Outer fold',fold,'choices locked and evaluated',flush=True)
    result=pd.DataFrame(predictions);result.to_csv(out/'out_of_fold_predictions.csv',index=False);pd.DataFrame(inner_scores).to_csv(out/'inner_scores.csv',index=False);pd.DataFrame(partitions).to_csv(out/'partitions.csv',index=False);write_json(out/'all_selection_locks.json',frozen)
    overall=[];per_class=[];per_fold=[];paired=[];group_table=[]
    base=result[result.method=='frozen_cnn'].set_index('structure_id').loc[frame.structure_id];base_pred=base.predicted_type.map(dict(zip(LABELS,range(k)))).to_numpy()
    for name,part in result.groupby('method'):
        part=part.set_index('structure_id').loc[frame.structure_id];pred=part.predicted_type.map(dict(zip(LABELS,range(k)))).to_numpy();m=metrics(y,pred,k);overall.append(dict(method=name,accuracy=float(m['accuracy']),macro_f1=float(m['macro_f1']),n_loci=n,n_groups=int(frame.group.nunique())))
        for c,label in enumerate(LABELS):per_class.append(dict(method=name,type=label,precision=float(m['precision'][c]),recall=float(m['recall'][c]),f1=float(m['f1'][c]),support=int(m['support'][c])))
        for f in range(cc['outer_folds']):
            sm=metrics(y[folds==f],pred[folds==f],k);per_fold.append(dict(method=name,fold=f,accuracy=float(sm['accuracy']),macro_f1=float(sm['macro_f1'])))
        for group in frame.group.unique():
            keep=frame.group.to_numpy()==group;sm=metrics(y[keep],pred[keep],k);group_table.append(dict(method=name,group=int(group),n=int(keep.sum()),accuracy=float(sm['accuracy']),macro_f1=float(sm['macro_f1'])))
        for reference in ['frozen_cnn','geometric_control']:
            if name==reference:continue
            b=result[result.method==reference].set_index('structure_id').loc[frame.structure_id].predicted_type.map(dict(zip(LABELS,range(k)))).to_numpy()
            for entry in paired_metric_intervals(y,pred,b,frame.group.to_numpy(),k,cfg['seed'],cfg['bootstrap_draws']):paired.append(dict(method=name,reference=reference,type=LABELS[int(entry['class_index'])] if entry['class_index'] is not None else 'all',**entry))
        part[part.true_type!=part.predicted_type].to_csv(out/f'errors_{name}.csv');pd.crosstab(pd.Categorical(frame.type,categories=LABELS),pd.Categorical(np.asarray(LABELS)[pred],categories=LABELS),dropna=False).to_csv(out/f'confusion_{name}.csv')
    pd.DataFrame(overall).to_csv(out/'overall.csv',index=False);pd.DataFrame(per_class).to_csv(out/'per_class.csv',index=False);pd.DataFrame(per_fold).to_csv(out/'per_fold.csv',index=False);pd.DataFrame(group_table).to_csv(out/'per_group.csv',index=False);pd.DataFrame(paired).to_csv(out/'paired_intervals.csv',index=False)
    write_json(out/'status.json',dict(completed_outer_folds=cc['outer_folds'],all_cases_retained=n,baseline_predictions_reproduced=True,maximum_contexts_disjoint=True,model_families=cc['method_families'],feature_control_not_extra_search=True,C_selected_by_training_only_three_inner_spatial_folds=True,probability_bias_selected_by_cyclic_inner_validation_only=True,frozen_CNN_not_retrained=True,independent_external_test=False,formal_choices_modified=False))
    print(pd.DataFrame(overall).to_string(index=False),flush=True)

if __name__=='__main__':main()

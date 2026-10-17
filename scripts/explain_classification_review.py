"""OOF sensitivity on annotated interval proxies vs matched-area/distance background."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from dlpro.method_review import load_config,write_json
from dlpro.experiment_core import LABELS,ShapeTransform,centered_indices
from dlpro.classification_models import FineScaleClassifier
from dlpro.classification_review import load_inputs,matched_masks
from experiment_classification_review import probability

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();cfg=load_config(a.config);root=Path(cfg['output']);out=root/'explanation';out.mkdir(exist_ok=True);assert not (out/'status.json').exists();torch.set_num_threads(cfg['threads'])
    frame,folds,raw,support=load_inputs();n=len(frame);cc=cfg['classifier'];k=len(LABELS);y=frame.type.map(dict(zip(LABELS,range(k)))).to_numpy();d=abs(np.arange(128)[:,None]-np.arange(128)[None,:]);rows=[];masks=[]
    for fold in range(cc['outer_folds']):
        test=folds==fold;train=(folds!=fold)&(folds!=(fold+1)%cc['outer_folds']);ids=np.flatnonzero(test);views=np.r_[ids,ids+n];tf=ShapeTransform.load(Path(cc['source'])/f'transform_fold{fold}.npz');allx=tf.transform(raw);x=allx[views];models=[]
        for seed in cc['seeds']:
            model=FineScaleClassifier();model.load_state_dict(torch.load(Path(cc['source'])/f'fine_consistent_seed{seed}_fold{fold}.pt',weights_only=True)['state_dict']);model.eval();models.append(model)
        original_view=probability(models,x);original=(original_view[:len(ids)]+original_view[len(ids):])/2;pred=original.argmax(1);competition=np.where(np.eye(k,dtype=bool)[y[ids]],-np.inf,original).argmax(1)
        median=np.stack([[np.median(allx[np.tile(train,2),channel][:,d==distance]) for distance in range(128)] for channel in range(3)])
        np.savez_compressed(out/f'training_fill_fold{fold}.npz',distance_median=median,neutral=np.clip(np.log(2)/tf.scale,0,1),training_indices=np.flatnonzero(train))
        for channel,bp in enumerate(cc['physical_bins_bp']):
            targets=[];controls=[];reasons=[]
            for i in ids:
                row=frame.iloc[i];start=(int(np.floor((row.center+bp/2)/bp))-64) if channel==0 else centered_indices(int(row.center))[channel-1]
                edges=(start+np.arange(128))*bp;inside=(edges<row.end)&(edges+bp>row.start);valid_rows=(support[i,channel]>=cc['support_fraction'])&(support[i+n,channel]>=cc['support_fraction']);valid=(d>=2)&valid_rows[:,None]&valid_rows[None,:]
                target,control=matched_masks(inside[:,None]&inside[None,:],valid,cfg['seed']+int(i)*len(cc['physical_bins_bp'])+channel)
                reason='available' if control is not None and target.any() else ('no_supported_proxy_pixels' if not target.any() else 'insufficient_background_at_matching_distances')
                targets.append(target);controls.append(np.zeros_like(target) if control is None else control);reasons.append(reason)
                masks.append(dict(structure_id=row.structure_id,fold=fold,group=int(row.group),type=row.type,bp=bp,proxy_pixels=int(target.sum()),background_pixels=0 if control is None else int(control.sum()),annotation_larger_than_half_view=bool((row.end-row.start)>bp*128*cfg['occlusion']['max_annotation_width_fraction']),status=reason))
            target=np.asarray(targets);control=np.asarray(controls);np.savez_compressed(out/f'masks_fold{fold}_{bp}bp.npz',structure_indices=ids,structure_proxy=target,background=control,status=np.asarray(reasons))
            for fill in cfg['occlusion']['fills']:
                value=0.0 if fill=='zero' else (float(np.clip(np.log(2)/tf.scale[channel],0,1)) if fill=='training_neutral' else median[channel,d])
                probabilities=[]
                for mask in [target,control]:
                    changed=x.copy();view_mask=np.concatenate([mask,mask]);changed[:,channel]=np.where(view_mask,value,changed[:,channel]);changed[:,:,d<2]=0;p=probability(models,changed);probabilities.append((p[:len(ids)]+p[len(ids):])/2)
                for local,i in enumerate(ids):
                    row=dict(structure_id=frame.structure_id.iloc[i],true_type=frame.type.iloc[i],group=int(frame.group.iloc[i]),fold=fold,bp=bp,fill=fill,original_prediction=LABELS[pred[local]],competing_class=LABELS[competition[local]],correct=bool(pred[local]==y[i]),status=reasons[local])
                    for label_index,label in enumerate(LABELS):
                        row[f'original_p_{label}']=float(original[local,label_index]);row[f'structure_drop_{label}']=float(original[local,label_index]-probabilities[0][local,label_index]) if reasons[local]=='available' else None;row[f'background_drop_{label}']=float(original[local,label_index]-probabilities[1][local,label_index]) if reasons[local]=='available' else None
                    for role,c in [('true',y[i]),('original_prediction',pred[local]),('competing',competition[local])]:
                        row[f'structure_drop_{role}']=float(original[local,c]-probabilities[0][local,c]) if reasons[local]=='available' else None;row[f'background_drop_{role}']=float(original[local,c]-probabilities[1][local,c]) if reasons[local]=='available' else None
                    row['true_structure_specificity']=row['structure_drop_true']-row['background_drop_true'] if reasons[local]=='available' else None;rows.append(row)
            print('Explanation outer fold',fold,'view',bp,'available',reasons.count('available'),'/',len(ids),flush=True)
    result=pd.DataFrame(rows);result.to_csv(out/'per_case.csv',index=False);pd.DataFrame(masks).to_csv(out/'mask_availability.csv',index=False)
    fields=['structure_drop_true','background_drop_true','true_structure_specificity','structure_drop_original_prediction','background_drop_original_prediction','structure_drop_competing','background_drop_competing']
    result.groupby(['true_type','bp','fill','correct'])[fields].agg(['mean','median','count']).to_csv(out/'class_summary.csv');result.groupby(['group','true_type','bp','fill'])[fields].mean().to_csv(out/'group_summary.csv')
    intervals=[]
    for key,part in result.groupby(['true_type','bp','fill']):
        use=part[part.status=='available'];groups=use.group.unique();value=use.true_structure_specificity.to_numpy();rng=np.random.default_rng(cfg['seed']);estimates=[]
        if len(groups):
            for _ in range(cfg['bootstrap_draws']):
                draw=rng.choice(groups,len(groups),replace=True);ix=np.concatenate([np.flatnonzero(use.group.to_numpy()==g) for g in draw]);estimates.append(float(value[ix].mean()))
        intervals.append(dict(zip(['type','bp','fill'],key),all_cases=len(part),available=len(use),unavailable=len(part)-len(use),groups=len(groups),mean_specificity=float(value.mean()) if len(value) else None,low=float(np.quantile(estimates,.025)) if estimates else None,high=float(np.quantile(estimates,.975)) if estimates else None))
    pd.DataFrame(intervals).to_csv(out/'specificity_intervals.csv',index=False)
    write_json(out/'status.json',dict(n_loci=n,per_case_rows=len(result),all_unavailable_retained=True,equal_area_and_exact_distance_histogram=True,common_native_support_required=True,fill_fitted_training_only=True,model_OOF=True,annotated_contact_pixel_segmentation_available=False,interval_square_is_only_proxy=True,background_mask_scattered_within_distance=True,not_causal_explanation=True,formal_choices_unchanged=True))
    print('Full OOF proxy/background sensitivity completed; no independent segmentation claim',flush=True)

if __name__=='__main__':main()

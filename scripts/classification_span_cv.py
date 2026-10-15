"""Auxiliary annotated-span target vs matched classifier-only continuation."""
import copy
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from sklearn.metrics import f1_score,classification_report
from dlpro.experiment_core import ShapeTransform,LABELS,configure,save_json
from dlpro.classification_models import SpanAwareClassifier,reverse_genome,repeat_js_divergence
out=Path('results/classification_span');out.mkdir(exist_ok=True);f=pd.read_csv('data/processed/known_windows.csv');folds=pd.read_csv('results/classification_cv/out_of_fold_predictions.csv').set_index('structure_id').loc[f.structure_id,'fold'].to_numpy();n=len(f);raw=np.concatenate([np.concatenate([np.load(f'data/processed/fine_rep{r}.npz')['raw'],np.load(f'data/processed/known_rep{r}.npz')['raw']],1) for r in [1,2]]);y=f.type.map(dict(zip(LABELS,range(3)))).to_numpy();logspan=np.log1p(f.length_bp.to_numpy());records=[];history_rows=[]
for fold in range(5):
 val=folds==(fold+1)%5;test=folds==fold;train=~(test|val);tf=ShapeTransform.load(f'results/classification_fine/transform_fold{fold}.npz');tx=torch.from_numpy(tf.transform(raw));mu=float(logspan[train].mean());sigma=float(logspan[train].std());target=(logspan-mu)/sigma;pools=[np.flatnonzero(train&(y==c)) for c in range(3)];ensembles={name:[] for name in ['matched_control','span_auxiliary']};span_ensemble=[]
 for recipe,coefficient in [('matched_control',0),('span_auxiliary',.1)]:
  for seed in [42,43,44]:
   configure(seed);model=SpanAwareClassifier();base=torch.load(f'results/classification_fine/fine_consistent_seed{seed}_fold{fold}.pt',weights_only=True);missing=model.load_state_dict(base['state_dict'],strict=False);assert set(missing.missing_keys)=={'span_head.weight','span_head.bias'} and not missing.unexpected_keys;optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.001);rng=np.random.default_rng(seed);best=(-1,float('-inf'));state=None;history=[]
   for epoch in range(15):
    model.train();losses=[];span_losses=[]
    for _ in range(int(np.ceil(2*train.sum()/30))):
     locations=np.concatenate([rng.choice(pool,5,replace=True) for pool in pools]);rng.shuffle(locations);ids=np.concatenate([locations,locations+n]);reverse=rng.random(len(locations))<.5;input=reverse_genome(tx[ids],torch.from_numpy(np.tile(reverse,2)));labels=torch.from_numpy(np.tile(y[locations],2)).long();span_target=torch.from_numpy(np.tile(target[locations],2).astype('float32'));optimizer.zero_grad();logits,predicted_span=model(input);span_loss=F.mse_loss(predicted_span,span_target);loss=F.cross_entropy(logits,labels,label_smoothing=.05)+.2*repeat_js_divergence(logits)+coefficient*span_loss;loss.backward();optimizer.step();losses.append(float(loss.detach()));span_losses.append(float(span_loss.detach()))
    model.eval()
    with torch.no_grad():prob=np.concatenate([F.softmax(model(z)[0],1).numpy() for z in tx.split(64)])
    paired=(prob[:n]+prob[n:])/2;vp=paired[val];vy=y[val];mf=float(f1_score(vy,vp.argmax(1),labels=[0,1,2],average='macro',zero_division=0));nll=float(-np.log(np.maximum(vp[np.arange(len(vy)),vy],1e-8)).mean());history.append(dict(epoch=epoch+1,objective=float(np.mean(losses)),normalized_span_mse=float(np.mean(span_losses)),val_macro_f1=mf,val_nll=nll))
    if (mf,-nll)>best:best=(mf,-nll);state=copy.deepcopy(model.state_dict());selected=epoch+1
   model.load_state_dict(state);model.eval()
   with torch.no_grad():outputs=[model(z) for z in tx.split(64)];prob=np.concatenate([F.softmax(v[0],1).numpy() for v in outputs]);spans=np.concatenate([v[1].numpy() for v in outputs])*sigma+mu
   pp=(prob[:n]+prob[n:])/2;paired_span=(spans[:n]+spans[n:])/2;ensembles[recipe].append(pp)
   if coefficient:span_ensemble.append(paired_span)
   torch.save(dict(state_dict=state,fold=fold,seed=seed,architecture='SpanAwareClassifier',span_weight=coefficient,span_log_mean=mu,span_log_std=sigma,selected_epoch=selected,initial_checkpoint=f'classification_fine/fine_consistent_seed{seed}_fold{fold}.pt'),out/f'{recipe}_seed{seed}_fold{fold}.pt');pd.DataFrame(history).to_csv(out/f'{recipe}_seed{seed}_fold{fold}_history.csv',index=False)
   for i in np.flatnonzero(test):
    row=dict(structure_id=f.structure_id.iloc[i],true_type=f.type.iloc[i],group=int(f.group.iloc[i]),fold=fold,recipe=f'{recipe}_seed{seed}',predicted_type=LABELS[pp[i].argmax()],predicted_log_span=float(paired_span[i]) if coefficient else None,true_log_span=float(logspan[i]));row.update({f'p_{label}':float(pp[i,c]) for c,label in enumerate(LABELS)});records.append(row)
   entry=dict(fold=fold,recipe=recipe,seed=seed,selected_epoch=selected,val_macro_f1=best[0],test_accuracy=float((pp[test].argmax(1)==y[test]).mean()),test_macro_f1=float(f1_score(y[test],pp[test].argmax(1),labels=[0,1,2],average='macro',zero_division=0)));history_rows.append(entry);print(entry,flush=True)
 for recipe,values in ensembles.items():
  pp=np.mean(values,0);span_prediction=np.mean(span_ensemble,0)
  for i in np.flatnonzero(test):
   row=dict(structure_id=f.structure_id.iloc[i],true_type=f.type.iloc[i],group=int(f.group.iloc[i]),fold=fold,recipe=f'{recipe}_ensemble',predicted_type=LABELS[pp[i].argmax()],predicted_log_span=float(span_prediction[i]) if recipe=='span_auxiliary' else None,true_log_span=float(logspan[i]));row.update({f'p_{label}':float(pp[i,c]) for c,label in enumerate(LABELS)});records.append(row)
frame=pd.DataFrame(records);frame.to_csv(out/'out_of_fold_predictions.csv',index=False);pd.DataFrame(history_rows).to_csv(out/'fold_metrics.csv',index=False);summary=[]
for recipe,g in frame.groupby('recipe'):
 assert len(g)==g.structure_id.nunique()==344;report=classification_report(g.true_type,g.predicted_type,labels=LABELS,output_dict=True,zero_division=0);save_json(out/f'{recipe}_report.json',report);summary.append(dict(recipe=recipe,accuracy=report['accuracy'],macro_f1=report['macro avg']['f1-score'],chid_recall=report['CHID']['recall'],chid_precision=report['CHID']['precision']))
pd.DataFrame(summary).to_csv(out/'summary.csv',index=False);span=frame[frame.recipe=='span_auxiliary_ensemble'];size_rows=[]
for label,g in span.groupby('true_type'):size_rows.append(dict(type=label,n_loci=len(g),log_span_rmse=float(np.sqrt(((g.predicted_log_span-g.true_log_span)**2).mean())),within_type_log_span_correlation=float(np.corrcoef(g.predicted_log_span,g.true_log_span)[0,1])))
pd.DataFrame(size_rows).to_csv(out/'span_metrics.csv',index=False);save_json(out/'protocol.json',dict(epochs=15,initialization='Per-fold corresponding-seed fixed fine_consistent checkpoints; no extra samples',seeds=[42,43,44],learning_rate=.0003,span_loss_weights=dict(matched_control=0,span_auxiliary=.1),span_target='log1p(annotation length) standardized from training loci only; never supplied at inference',loss='Balanced CE smoothing.05 +.2pairedJS +coefficient*normalized span MSE',selection='Validation macroF1 then NLL chooses checkpoint, never held-out span or class metrics',primary='span_auxiliary_ensemble',matched_control='Identical architecture, initialization, optimization schedule and15epochs, auxiliary loss weight0',limitations=['Retrospective same-study grouped CV, repeatedly observed test positions','Only26CHID loci and nested label ambiguity','Auxiliary span predictions can reflect category priors rather than actual boundaries; within-class correlations retained','Source checkpoints already selected on same validation fold; not an independent confirmatory experiment']))
print(pd.DataFrame(summary).to_string(index=False),flush=True)

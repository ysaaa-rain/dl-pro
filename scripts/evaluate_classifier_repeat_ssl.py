"""Evaluate fixed self-supervision arms only after all fold decisions are saved."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report
from dlpro.experiment_core import LABELS,ShapeTransform,configure,save_json
from dlpro.classification_models import FineScaleClassifier
out=Path('results/classification_repeat_ssl');decisions=json.loads((out/'decisions.json').read_text());assert len(decisions['folds'])==5;decision_hash=hashlib.sha256((out/'decisions.json').read_bytes()).hexdigest();configure(42);known=pd.read_csv('data/processed/known_windows.csv');folds=pd.read_csv('results/classification_cv/out_of_fold_predictions.csv').set_index('structure_id').loc[known.structure_id,'fold'].to_numpy();n=len(known);raw=np.concatenate([np.concatenate([np.load(f'data/processed/fine_rep{r}.npz')['raw'],np.load(f'data/processed/known_rep{r}.npz')['raw']],1) for r in [1,2]]);records=[]
for decision in decisions['folds']:
 fold=decision['fold'];ids=np.flatnonzero(folds==fold);view_ids=np.concatenate([ids,ids+n]);tx=torch.from_numpy(ShapeTransform.load(out/f'transform_fold{fold}.npz').transform(raw[view_ids]));ensemble={}
 for recipe in ['matched_control','repeat_pretrained']:
  predictions=[]
  for seed in [42,43,44]:
   cp=torch.load(out/f'{recipe}_seed{seed}_fold{fold}.pt',weights_only=True);assert cp['protocol_sha256']==decisions['protocol_sha256'];model=FineScaleClassifier();model.load_state_dict(cp['state_dict']);model.eval()
   with torch.no_grad():p=torch.cat([model(batch).softmax(1) for batch in tx.split(64)]).numpy()
   paired=(p[:len(ids)]+p[len(ids):])/2;predictions.append(paired)
  original=np.mean(predictions,0);factor=decision['control_chid_factor' if recipe=='matched_control' else 'ssl_chid_factor'];calibrated=original*np.array([factor,1,1]);calibrated/=calibrated.sum(1,keepdims=True);ensemble[recipe]=calibrated
  for variant,prob in [(recipe+'_raw',original),(recipe+'_calibrated',calibrated)]:
   for local,i in enumerate(ids):
    row=dict(structure_id=known.structure_id.iloc[i],true_type=known.type.iloc[i],group=int(known.group.iloc[i]),fold=fold,recipe=variant,predicted_type=LABELS[prob[local].argmax()]);row.update({f'p_{label}':float(prob[local,k]) for k,label in enumerate(LABELS)});records.append(row)
 for local,i in enumerate(ids):
  prob=ensemble[decision['selected_recipe']][local];row=dict(structure_id=known.structure_id.iloc[i],true_type=known.type.iloc[i],group=int(known.group.iloc[i]),fold=fold,recipe='validation_selected',selected_recipe=decision['selected_recipe'],predicted_type=LABELS[prob.argmax()]);row.update({f'p_{label}':float(prob[k]) for k,label in enumerate(LABELS)});records.append(row)
frame=pd.DataFrame(records);frame.to_csv(out/'out_of_fold_predictions.csv',index=False);summary=[]
for recipe,g in frame.groupby('recipe'):
 assert len(g)==g.structure_id.nunique()==344;report=classification_report(g.true_type,g.predicted_type,labels=LABELS,output_dict=True,zero_division=0);save_json(out/f'{recipe}_metrics.json',report);summary.append(dict(recipe=recipe,accuracy=report['accuracy'],macro_f1=report['macro avg']['f1-score'],chid_precision=report['CHID']['precision'],chid_recall=report['CHID']['recall'],chid_f1=report['CHID']['f1-score']))
pd.DataFrame(summary).to_csv(out/'summary.csv',index=False);save_json(out/'evaluation_protocol.json',dict(decisions_sha256=decision_hash,protocol_sha256=decisions['protocol_sha256'],test_evaluated_after_all_five_selections=True,original_labels_unchanged=True,limitations=['Same previously exposed groups, retrospective comparison','Validation gate protects otherclassF1 on validation only; not a test guarantee','Closed-set image labels include nested/overlapping biological annotations','No external independent test set']))
print(pd.DataFrame(summary).to_string(index=False),flush=True)

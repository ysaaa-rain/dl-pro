"""Validation-only CHID gate over a small-sample image kernel classifier."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from sklearn.metrics import f1_score,classification_report
from dlpro.experiment_core import ShapeTransform,Classifier,LABELS,configure,save_json
configure(42);out=Path('results/classification_hybrid');out.mkdir(exist_ok=True)
f=pd.read_csv('data/processed/known_windows.csv');old=pd.read_csv('results/classification_cv/out_of_fold_predictions.csv').set_index('structure_id');folds=old.loc[f.structure_id,'fold'].to_numpy();raw=np.concatenate([np.load(f'data/processed/known_rep{r}.npz')['raw'] for r in [1,2]]);n=len(f);y=f.type.map(dict(zip(LABELS,range(3)))).to_numpy();grid=pd.read_csv('results/classification_kernel/validation_grid.csv');records=[];choices=[]
for fold in range(5):
 val=folds==(fold+1)%5;test=folds==fold;train=np.tile(~(test|val),2);tf=ShapeTransform.load(f'results/classification_kernel/transform_fold{fold}.npz');x=tf.transform(raw);p=np.load(f'results/classification_kernel/pca_fold{fold}.npz');features=torch.einsum('nd,kd->nk',torch.from_numpy(x.reshape(2*n,-1))-torch.from_numpy(p['mean']),torch.from_numpy(p['components'])).numpy().astype(float)/np.maximum(p['scale'],1e-6);distance=np.maximum(np.einsum('nd,nd->n',features,features)[:,None]+np.einsum('nd,nd->n',p['training_embeddings'],p['training_embeddings'])[None]-2*np.einsum('nd,md->nm',features,p['training_embeddings'],optimize=False),0)
 parameter=max(grid[grid.fold.eq(fold)].to_dict('records'),key=lambda r:(r['val_macro_f1'],r['val_balanced_accuracy'],-r['C'],-r['gamma_factor']));svm=joblib.load(f'results/classification_kernel/svm_fold{fold}.joblib');decision=svm.decision_function(np.exp(-parameter['gamma_factor']/32*distance));kernel=(decision[:n]+decision[n:])/2
 # The imbalance head is frozen as balanced_shared before threshold evaluation.
 head=Classifier();head.load_state_dict(torch.load(f'results/classification_balanced/balanced_shared_fold{fold}.pt',weights_only=True)['state_dict']);head.eval()
 with torch.no_grad():prob=np.concatenate([F.softmax(head(z),1).numpy() for z in torch.from_numpy(x).split(64)])
 probability=(prob[:n]+prob[n:])/2;best=(-1,float('-inf'));trials=[]
 for threshold in [.3,.4,.5,.6,.7,.8]:
  pred=np.where(probability[:,0]>=threshold,0,kernel[:,1:].argmax(1)+1);mf=float(f1_score(y[val],pred[val],labels=[0,1,2],average='macro',zero_division=0));quality=(mf,threshold);trials.append(dict(threshold=threshold,val_macro_f1=mf))
  if quality>best:best=quality;selected=pred.copy();selected_threshold=threshold
 choices.append(dict(fold=fold,threshold=selected_threshold,val_macro_f1=best[0],validation_grid=trials))
 for i in np.flatnonzero(test):records.append(dict(structure_id=f.structure_id.iloc[i],true_type=f.type.iloc[i],group=int(f.group.iloc[i]),fold=fold,predicted_type=LABELS[selected[i]],chid_gate_score=float(probability[i,0]),threshold=selected_threshold,rest_kernel_label=LABELS[kernel[i,1:].argmax()+1]))
 print('hybrid fold',fold,'threshold',selected_threshold,'valmacro',best[0],flush=True)
frame=pd.DataFrame(records);assert len(frame)==frame.structure_id.nunique()==344;frame.to_csv(out/'out_of_fold_predictions.csv',index=False);report=classification_report(frame.true_type,frame.predicted_type,labels=LABELS,output_dict=True,zero_division=0);save_json(out/'classification_report.json',report);save_json(out/'protocol.json',dict(seed=42,head='balanced_shared CNN trained on outer training loci',rest_classifier='Image PCA RBF SVM trained on same outer training loci',fold_choices=choices,selection='Only cyclic validation macroF1 selects CHID threshold0.3..0.8; ties prefer larger threshold',limitations=['Retrospective existing test positions; not an unbiased model-search final benchmark','Only26CHID examples; gate threshold can vary across folds','Kernel OVR scores are not probabilities; CHID gate not clinically or externally calibrated']))
print({k:report[k] for k in ['accuracy','macro avg','CHID']},flush=True)

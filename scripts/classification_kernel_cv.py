"""Training-only image PCA and balanced RBF SVM with grouped validation selection."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch,joblib
from sklearn.svm import SVC
from sklearn.metrics import f1_score,balanced_accuracy_score,classification_report
from dlpro.experiment_core import ShapeTransform,LABELS,configure,save_json
configure(42);out=Path('results/classification_kernel');out.mkdir(exist_ok=True)
f=pd.read_csv('data/processed/known_windows.csv');old=pd.read_csv('results/classification_cv/out_of_fold_predictions.csv').set_index('structure_id');folds=old.loc[f.structure_id,'fold'].to_numpy();raw=np.concatenate([np.load(f'data/processed/known_rep{r}.npz')['raw'] for r in [1,2]]);n=len(f);y0=f.type.map(dict(zip(LABELS,range(3)))).to_numpy();y=np.tile(y0,2);records=[];grids=[]
for fold in range(5):
 test=folds==fold;val=folds==(fold+1)%5;train=np.tile(~(test|val),2);tf=ShapeTransform().fit(raw[train]);tf.save(out/f'transform_fold{fold}.npz');x=tf.transform(raw).reshape(2*n,-1);xt=torch.from_numpy(x);mean=xt[train].mean(0);u,s,v=torch.pca_lowrank(xt[train]-mean,q=32,center=False,niter=3);features=torch.einsum('nd,dk->nk',xt-mean,v).numpy().astype('float64');std=features[train].std(0);features/=np.maximum(std,1e-6)
 distance=np.maximum(np.einsum('nd,nd->n',features,features)[:,None]+np.einsum('nd,nd->n',features[train],features[train])[None]-2*np.einsum('nd,md->nm',features,features[train],optimize=False),0);best=(-1,-1,float('-inf'),float('-inf'))
 for C in [.1,1,10]:
  for factor in [.25,1]:
   gamma=factor/32;kernel=np.exp(-gamma*distance);model=SVC(C=C,kernel='precomputed',class_weight='balanced',decision_function_shape='ovr');model.fit(kernel[train],y[train]);decision=model.decision_function(kernel);paired=(decision[:n]+decision[n:])/2;pred=paired.argmax(1);mf=float(f1_score(y0[val],pred[val],labels=[0,1,2],average='macro',zero_division=0));ba=float(balanced_accuracy_score(y0[val],pred[val]));quality=(mf,ba,-C,-factor);grids.append(dict(fold=fold,C=C,gamma_factor=factor,val_macro_f1=mf,val_balanced_accuracy=ba))
   if quality>best:best=quality;selected=model;selected_pred=pred;selected_decision=paired;selected_C=C;selected_factor=factor
 np.savez_compressed(out/f'pca_fold{fold}.npz',mean=mean.numpy(),components=v.numpy().T,scale=std,training_embeddings=features[train]);joblib.dump(selected,out/f'svm_fold{fold}.joblib',compress=3)
 for i in np.flatnonzero(test):
  r=dict(structure_id=f.structure_id.iloc[i],true_type=f.type.iloc[i],group=int(f.group.iloc[i]),fold=fold,predicted_type=LABELS[selected_pred[i]]);r.update({f'decision_{label}':float(selected_decision[i,c]) for c,label in enumerate(LABELS)});records.append(r)
 print('kernel fold',fold,'C',selected_C,'gamma factor',selected_factor,'valmacro',best[0],flush=True)
frame=pd.DataFrame(records);assert frame.structure_id.nunique()==len(frame)==344;frame.to_csv(out/'out_of_fold_predictions.csv',index=False);pd.DataFrame(grids).to_csv(out/'validation_grid.csv',index=False);report=classification_report(frame.true_type,frame.predicted_type,labels=LABELS,output_dict=True,zero_division=0);save_json(out/'classification_report.json',report);save_json(out/'protocol.json',dict(seed=42,pca_rank=32,preprocessing='ShapeTransform and PCA/whitening fitted on training rows only per fold',kernel='RBF on whitened image PCA; distances computed by independent einsum, no BLAS matmul',validation_grid_C=[.1,1,10],validation_gamma_factor=[.25,1],selection='Validation macroF1 then balanced accuracy then smaller C/gamma',unit='Average OVR decision scores of the two repeats per location; scores are not probabilities',limitations=['Retrospective same-study grouped CV, previously observed test positions','Only26CHID labels and nested categories','No annotated length or boundary metadata input']))
print({k:report[k] for k in ['accuracy','macro avg','CHID']},flush=True)

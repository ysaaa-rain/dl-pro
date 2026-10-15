"""Grouped outer folds, validation-only recipe selection, balanced sampling ablation."""
import copy
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from sklearn.metrics import f1_score,classification_report,confusion_matrix
from dlpro.experiment_core import Classifier,ShapeTransform,LABELS,configure,save_json
from dlpro.classification_models import ScaleAwareClassifier,reverse_genome

out=Path('results/classification_balanced');out.mkdir(exist_ok=True)
f=pd.read_csv('data/processed/known_windows.csv');fold_table=pd.read_csv('results/classification_cv/out_of_fold_predictions.csv').set_index('structure_id');folds=fold_table.loc[f.structure_id,'fold'].to_numpy();n=len(f);raw=np.concatenate([np.load(f'data/processed/known_rep{r}.npz')['raw'] for r in [1,2]]);y0=f.type.map(dict(zip(LABELS,range(3)))).to_numpy();y=np.tile(y0,2);ty=torch.from_numpy(y).long();recipes={'balanced_shared':Classifier,'balanced_multiscale':ScaleAwareClassifier};predictions=[];choices=[];records=[]
for fold in range(5):
    test=folds==fold;val=folds==(fold+1)%5;train0=~(test|val);train=np.tile(train0,2);tf=ShapeTransform().fit(raw[train]);tf.save(out/f'transform_fold{fold}.npz');tx=torch.from_numpy(tf.transform(raw));assert set(f.loc[train0,'group']).isdisjoint(set(f.loc[test|val,'group']))
    validation={};probabilities={}
    for recipe,constructor in recipes.items():
        configure(42);model=constructor();optim=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001);rng=np.random.default_rng(42);pools=[np.flatnonzero(train&(y==c)) for c in range(3)];best=(-1,float('-inf'));state=None;history=[]
        for epoch in range(35):
            model.train();losses=[]
            for batch in range(int(np.ceil(train.sum()/30))):
                ids=np.concatenate([rng.choice(pool,10,replace=True) for pool in pools]);rng.shuffle(ids);input=reverse_genome(tx[ids],torch.from_numpy(rng.random(len(ids))<.5));optim.zero_grad();loss=F.cross_entropy(model(input),ty[ids],label_smoothing=.05);loss.backward();optim.step();losses.append(float(loss.detach()))
            model.eval()
            with torch.no_grad():prob=np.concatenate([F.softmax(model(z),1).numpy() for z in tx.split(64)])
            paired=(prob[:n]+prob[n:])/2;vp=paired[val];vy=y0[val];mf=float(f1_score(vy,vp.argmax(1),labels=[0,1,2],average='macro',zero_division=0));nll=float(-np.log(np.maximum(vp[np.arange(len(vy)),vy],1e-8)).mean());history.append(dict(epoch=epoch+1,loss=float(np.mean(losses)),val_macro_f1=mf,val_nll=nll))
            if (mf,-nll)>best:best=(mf,-nll);state=copy.deepcopy(model.state_dict());selected=epoch+1
        model.load_state_dict(state);model.eval()
        with torch.no_grad():prob=np.concatenate([F.softmax(model(z),1).numpy() for z in tx.split(64)])
        pp=(prob[:n]+prob[n:])/2;probabilities[recipe]=pp;validation[recipe]=best
        torch.save(dict(state_dict=state,architecture=constructor.__name__,seed=42,fold=fold,selected_epoch=selected,recipe=recipe),out/f'{recipe}_fold{fold}.pt');pd.DataFrame(history).to_csv(out/f'{recipe}_history_fold{fold}.csv',index=False)
        for i in np.flatnonzero(test):
            row=dict(structure_id=f.structure_id.iloc[i],true_type=f.type.iloc[i],group=int(f.group.iloc[i]),fold=fold,recipe=recipe,predicted_type=LABELS[pp[i].argmax()]);row.update({f'p_{label}':float(pp[i,c]) for c,label in enumerate(LABELS)});predictions.append(row)
        record=dict(fold=fold,recipe=recipe,selected_epoch=selected,val_macro_f1=best[0],test_accuracy=float((pp[test].argmax(1)==y0[test]).mean()),test_macro_f1=float(f1_score(y0[test],pp[test].argmax(1),labels=[0,1,2],average='macro',zero_division=0)));records.append(record);print(record,flush=True)
    choice=max(recipes,key=lambda name:validation[name]);choices.append(dict(fold=fold,chosen_recipe=choice,val_macro_f1=validation[choice][0],val_nll=-validation[choice][1]));pp=probabilities[choice]
    for i in np.flatnonzero(test):
        row=dict(structure_id=f.structure_id.iloc[i],true_type=f.type.iloc[i],group=int(f.group.iloc[i]),fold=fold,recipe='validation_selected',selected_recipe=choice,predicted_type=LABELS[pp[i].argmax()]);row.update({f'p_{label}':float(pp[i,c]) for c,label in enumerate(LABELS)});predictions.append(row)
frame=pd.DataFrame(predictions);frame.to_csv(out/'out_of_fold_predictions.csv',index=False);pd.DataFrame(records).to_csv(out/'fold_metrics.csv',index=False);pd.DataFrame(choices).to_csv(out/'recipe_choices.csv',index=False)
summary=[]
for recipe,g in frame.groupby('recipe'):
    assert len(g)==344 and g.structure_id.nunique()==344;report=classification_report(g.true_type,g.predicted_type,labels=LABELS,output_dict=True,zero_division=0);save_json(out/f'{recipe}_report.json',report);pd.DataFrame(confusion_matrix(g.true_type,g.predicted_type,labels=LABELS),index=LABELS,columns=LABELS).to_csv(out/f'{recipe}_confusion.csv');summary.append(dict(recipe=recipe,accuracy=report['accuracy'],macro_f1=report['macro avg']['f1-score'],chid_recall=report['CHID']['recall'],chid_precision=report['CHID']['precision']))
pd.DataFrame(summary).to_csv(out/'summary.csv',index=False)
save_json(out/'protocol.json',dict(seed=42,outer_folds=5,epochs=35,split_source='Same immutable overlap groups and outer folds as classification_cv',validation='Next cyclic fold; every model and recipe chosen only on validation macroF1 then NLL',sampling='Each batch30:10 examples per class, with replacement from training only; two repeats share locus split',augmentation='Reverse both genome axes with probability0.5',loss='Unweighted cross entropy with label smoothing0.05; no extra class weights on balanced batches',recipes=list(recipes),normalization='Training-only ShapeTransform per outer fold; no length/type metadata',limitations=['Existing outer test positions were examined previously; retrospective comparison','Validation recipe choice is internal; no independent source validation','Repeats share training locations; not cross-replicate training isolation','Only26CHID loci, metrics and model choice may be unstable']))
print(pd.DataFrame(summary).to_string(index=False),flush=True)

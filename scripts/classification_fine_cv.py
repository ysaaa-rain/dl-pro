"""Three resolution views and paired-repeat consistency; fixed spatial outer folds."""
import copy
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from sklearn.metrics import f1_score,classification_report
from dlpro.experiment_core import ShapeTransform,LABELS,configure,save_json
from dlpro.classification_models import FineScaleClassifier,reverse_genome,repeat_js_divergence

out=Path('results/classification_fine');out.mkdir(exist_ok=True)
f=pd.read_csv('data/processed/known_windows.csv');fold_table=pd.read_csv('results/classification_cv/out_of_fold_predictions.csv').set_index('structure_id');folds=fold_table.loc[f.structure_id,'fold'].to_numpy();n=len(f)
raw=np.concatenate([np.concatenate([np.load(f'data/processed/fine_rep{r}.npz')['raw'],np.load(f'data/processed/known_rep{r}.npz')['raw']],axis=1) for r in [1,2]])
y0=f.type.map(dict(zip(LABELS,range(3)))).to_numpy();recipes=[('fine_multiscale',0,42),('fine_consistent',.2,42),('fine_consistent',.2,43),('fine_consistent',.2,44)];records=[];fold_rows=[]
for fold in range(5):
    test=folds==fold;val=folds==(fold+1)%5;train=~(test|val);tf=ShapeTransform().fit(raw[np.tile(train,2)]);tf.save(out/f'transform_fold{fold}.npz');tx=torch.from_numpy(tf.transform(raw));pools=[np.flatnonzero(train&(y0==c)) for c in range(3)];ensemble=[]
    for recipe,coefficient,seed in recipes:
        configure(seed);model=FineScaleClassifier();optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001);rng=np.random.default_rng(seed);best=(-1,float('-inf'));state=None;history=[]
        for epoch in range(35):
            model.train();losses=[];consistency=[]
            for _ in range(int(np.ceil(2*train.sum()/30))):
                locations=np.concatenate([rng.choice(pool,5,replace=True) for pool in pools]);rng.shuffle(locations);ids=np.concatenate([locations,locations+n]);reverse=rng.random(len(locations))<.5;input=reverse_genome(tx[ids],torch.from_numpy(np.tile(reverse,2)));labels=torch.from_numpy(np.tile(y0[locations],2)).long();optimizer.zero_grad();logits=model(input);js=repeat_js_divergence(logits);loss=F.cross_entropy(logits,labels,label_smoothing=.05)+coefficient*js;loss.backward();optimizer.step();losses.append(float(loss.detach()));consistency.append(float(js.detach()))
            model.eval()
            with torch.no_grad():prob=np.concatenate([F.softmax(model(z),1).numpy() for z in tx.split(64)])
            paired=(prob[:n]+prob[n:])/2;vp=paired[val];vy=y0[val];mf=float(f1_score(vy,vp.argmax(1),labels=[0,1,2],average='macro',zero_division=0));nll=float(-np.log(np.maximum(vp[np.arange(len(vy)),vy],1e-8)).mean());history.append(dict(epoch=epoch+1,loss=float(np.mean(losses)),repeat_js=float(np.mean(consistency)),val_macro_f1=mf,val_nll=nll))
            if (mf,-nll)>best:best=(mf,-nll);state=copy.deepcopy(model.state_dict());selected=epoch+1
        model.load_state_dict(state);model.eval()
        with torch.no_grad():prob=np.concatenate([F.softmax(model(z),1).numpy() for z in tx.split(64)])
        pp=(prob[:n]+prob[n:])/2
        if recipe=='fine_consistent':ensemble.append(pp)
        torch.save(dict(state_dict=state,architecture='FineScaleClassifier',seed=seed,fold=fold,selected_epoch=selected,repeat_js_coefficient=coefficient),out/f'{recipe}_seed{seed}_fold{fold}.pt');pd.DataFrame(history).to_csv(out/f'{recipe}_seed{seed}_fold{fold}_history.csv',index=False)
        name=f'{recipe}_seed{seed}'
        for i in np.flatnonzero(test):
            row=dict(structure_id=f.structure_id.iloc[i],true_type=f.type.iloc[i],group=int(f.group.iloc[i]),fold=fold,recipe=name,predicted_type=LABELS[pp[i].argmax()]);row.update({f'p_{label}':float(pp[i,c]) for c,label in enumerate(LABELS)});records.append(row)
        entry=dict(fold=fold,recipe=name,selected_epoch=selected,val_macro_f1=best[0],test_accuracy=float((pp[test].argmax(1)==y0[test]).mean()),test_macro_f1=float(f1_score(y0[test],pp[test].argmax(1),labels=[0,1,2],average='macro',zero_division=0)));fold_rows.append(entry);print(entry,flush=True)
    pp=np.mean(ensemble,0)
    for i in np.flatnonzero(test):
        row=dict(structure_id=f.structure_id.iloc[i],true_type=f.type.iloc[i],group=int(f.group.iloc[i]),fold=fold,recipe='fine_consistent_ensemble',predicted_type=LABELS[pp[i].argmax()]);row.update({f'p_{label}':float(pp[i,c]) for c,label in enumerate(LABELS)});records.append(row)
frame=pd.DataFrame(records);frame.to_csv(out/'out_of_fold_predictions.csv',index=False);pd.DataFrame(fold_rows).to_csv(out/'fold_metrics.csv',index=False);summary=[]
for recipe,g in frame.groupby('recipe'):
    assert len(g)==344 and g.structure_id.nunique()==344;report=classification_report(g.true_type,g.predicted_type,labels=LABELS,output_dict=True,zero_division=0);save_json(out/f'{recipe}_report.json',report);summary.append(dict(recipe=recipe,accuracy=report['accuracy'],macro_f1=report['macro avg']['f1-score'],chid_recall=report['CHID']['recall'],chid_precision=report['CHID']['precision']))
pd.DataFrame(summary).to_csv(out/'summary.csv',index=False);save_json(out/'protocol.json',dict(epochs=35,primary='fine_consistent_ensemble',seeds=[42,43,44],ablation='fine_multiscale_seed42 coefficient0 vs fine_consistent_seed42 coefficient0.2',scales_bp=[50,100,200],widths_bp=[6400,12800,25600],split='Existing maximum25.6kb overlap groups and outer folds unchanged;50bp view lies entirely within original extent',sampling='Five training locations per class per batch; both repeats with shared genome reversal',normalization='Three-channel transform fit only on outer training locations',loss='Balanced CE smoothing0.05 + coefficient*paired Jensen-Shannon divergence',selection='Epoch chosen on cyclic validation macroF1 then NLL; no test selection; fixed three-seed average',limitations=['Retrospective existing test locations, not independent external validation','Only26CHID labels and overlapping/nested structures','Both repeats are fitted at training locations; test is spatially held out','No annotated boundaries or lengths as model inputs']))
print(pd.DataFrame(summary).to_string(index=False),flush=True)

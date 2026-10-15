"""Small-sample geometric features vs CNN; validation-only depth and model selection."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,classification_report
from dlpro.experiment_core import ShapeTransform,Classifier,LABELS,configure,save_json
from dlpro.shape_features import summarize_shapes
configure(42);out=Path('results/classification_shape');out.mkdir(exist_ok=True)
f=pd.read_csv('data/processed/known_windows.csv');old=pd.read_csv('results/classification_cv/out_of_fold_predictions.csv').set_index('structure_id');folds=old.loc[f.structure_id,'fold'].to_numpy();raw=np.concatenate([np.load(f'data/processed/known_rep{r}.npz')['raw'] for r in [1,2]]);n=len(f);y0=f.type.map(dict(zip(LABELS,range(3)))).to_numpy();y=np.tile(y0,2);predictions=[];choices=[];summary=[]

def score(prob,truth):
    return (float(f1_score(truth,prob.argmax(1),labels=[0,1,2],average='macro',zero_division=0)),float(np.log(np.maximum(prob[np.arange(len(truth)),truth],1e-8)).mean()))
for fold in range(5):
    test=folds==fold;val=folds==(fold+1)%5;train=np.tile(~(test|val),2);tf=ShapeTransform().fit(raw[train]);tf.save(out/f'transform_fold{fold}.npz');x=tf.transform(raw);features=summarize_shapes(x);np.savez_compressed(out/f'features_fold{fold}.npz',values=features);best=(-1,float('-inf'))
    grid=[]
    for depth in [3,6,None]:
        model=RandomForestClassifier(n_estimators=300,max_depth=depth,min_samples_leaf=4,class_weight='balanced_subsample',max_features='sqrt',random_state=42,n_jobs=4);model.fit(features[train],y[train]);prob=model.predict_proba(features);paired=(prob[:n]+prob[n:])/2;s=score(paired[val],y0[val]);grid.append(dict(depth=depth,val_macro_f1=s[0],val_nll=-s[1]))
        if s>best:best=s;selected=model;forest_prob=paired;selected_depth=depth
    # Portable tree arrays plus an explicit feature recipe; pickle is not required for inference.
    trees={}
    for i,estimator in enumerate(selected.estimators_):
        t=estimator.tree_;trees.update({f'{i}_left':t.children_left,f'{i}_right':t.children_right,f'{i}_feature':t.feature,f'{i}_threshold':t.threshold,f'{i}_value':t.value})
    np.savez_compressed(out/f'forest_fold{fold}.npz',**trees)
    cnn=Classifier();path='results/classification/model_seed42.pt' if fold==0 else f'results/classification_cv/model_fold{fold}.pt';cnn.load_state_dict(torch.load(path,weights_only=True)['state_dict']);cnn.eval()
    with torch.no_grad():prob=np.concatenate([F.softmax(cnn(z),1).numpy() for z in torch.from_numpy(x).split(64)])
    cnn_prob=(prob[:n]+prob[n:])/2;cnn_score=score(cnn_prob[val],y0[val]);choice='geometric_forest' if best>cnn_score else 'original_cnn';selected_prob=forest_prob if choice=='geometric_forest' else cnn_prob
    choices.append(dict(fold=fold,selected_depth=selected_depth,validation_grid=grid,forest_val_macro_f1=best[0],cnn_val_macro_f1=cnn_score[0],choice=choice))
    for name,pp in [('geometric_forest',forest_prob),('validation_selected',selected_prob)]:
        for i in np.flatnonzero(test):
            r=dict(structure_id=f.structure_id.iloc[i],true_type=f.type.iloc[i],group=int(f.group.iloc[i]),fold=fold,recipe=name,predicted_type=LABELS[pp[i].argmax()]);r.update({f'p_{l}':float(pp[i,c]) for c,l in enumerate(LABELS)});predictions.append(r)
    print('shape fold',fold,'depth',selected_depth,'choice',choice,'forest val',best[0],'CNN val',cnn_score[0],flush=True)
frame=pd.DataFrame(predictions);frame.to_csv(out/'out_of_fold_predictions.csv',index=False)
for recipe,g in frame.groupby('recipe'):
    assert len(g)==344 and g.structure_id.nunique()==344;report=classification_report(g.true_type,g.predicted_type,labels=LABELS,output_dict=True,zero_division=0);save_json(out/f'{recipe}_report.json',report);summary.append(dict(recipe=recipe,accuracy=report['accuracy'],macro_f1=report['macro avg']['f1-score'],chid_recall=report['CHID']['recall'],chid_precision=report['CHID']['precision']))
pd.DataFrame(summary).to_csv(out/'summary.csv',index=False);save_json(out/'protocol.json',dict(seed=42,n_trees=300,min_leaf_samples=4,fold_choices=choices,feature_recipe='Training O/E image full/central4x4 means/std; center radii8/16/32/48 and off-diagonal distance2/8/16/32/64 mean/std/90th quantile, both physical scales',no_annotation_length_or_boundary_inputs=True,validation='Depth3/6/unlimited and forest vs originalCNN selected only on cyclic validation fold',limitations=['Existing test locations already observed; retrospective same-study CV','Nested CHID/CHIN annotations can produce nearly indistinguishable centered images','Label metadata never used as model input','Balanced neural and forest recipes evaluated separately; no test-score choice of final model']))
print(pd.DataFrame(summary).to_string(index=False),flush=True)

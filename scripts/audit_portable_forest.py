"""Reproduce every forest OOF probability without sklearn tree prediction."""
import numpy as np
import pandas as pd
from dlpro.experiment_core import LABELS,save_json
from dlpro.shape_features import forest_probabilities
known=pd.read_csv('data/processed/known_windows.csv');n=len(known);pred=pd.read_csv('results/classification_shape/out_of_fold_predictions.csv');pred=pred[pred.recipe=='geometric_forest'];errors=[]
for fold in range(5):
 values=np.load(f'results/classification_shape/features_fold{fold}.npz')['values'];prob=forest_probabilities(np.load(f'results/classification_shape/forest_fold{fold}.npz'),values);paired=(prob[:n]+prob[n:])/2;rows=pred[pred.fold==fold];ids=known.set_index('structure_id').index.get_indexer(rows.structure_id);error=float(abs(paired[ids]-rows[[f'p_{v}' for v in LABELS]].to_numpy()).max());assert error<1e-12;errors.append(error)
save_json('results/model_validation/portable_forest_checks.json',dict(max_probability_error=max(errors),all_344_oof_probabilities_reproduced=True));print(errors)

"""Reproduce all344 frozen OOF probabilities and probe a circular raw coordinate."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from dlpro.annotation_tool import AnnotationEngine
from dlpro.experiment_core import save_json
root=Path(__file__).resolve().parents[1];engine=AnnotationEngine(root);known=engine.known;n=len(known);raw=np.stack([np.concatenate([np.load(root/f'data/processed/fine_rep{r}.npz')['raw'],np.load(root/f'data/processed/known_rep{r}.npz')['raw']],axis=1) for r in [1,2]])
expected=pd.read_csv(root/'results/classification_fine/out_of_fold_predictions.csv');expected=expected[expected.recipe=='fine_consistent_ensemble'].set_index('structure_id');errors=[]
for fold in range(5):
 ids=np.flatnonzero(engine.folds==fold);models,transform=engine.load(fold);x=torch.from_numpy(transform.transform(raw[:,ids].reshape(-1,3,128,128)))
 with torch.no_grad():prob=torch.stack([torch.cat([m(b).softmax(-1) for b in x.split(64)]) for m in models]).mean(0).reshape(2,len(ids),3).mean(0).numpy()
 truth=expected.loc[known.structure_id.iloc[ids],['p_CHID','p_CHIN','p_OPCID']].to_numpy();errors.append(float(abs(prob-truth).max()))
assert max(errors)<1e-6
live=engine.predict(12345);assert live['input_source']=='raw_cool' and live['view_intervals'][2]['wraps'];assert np.isfinite(live['views']).all() and np.isfinite(live['saliency']).all();assert abs(sum(live['probabilities'].values())-1)<1e-6
save_json(root/'results/annotation_tool/validation.json',dict(n_oof_loci=n,max_probability_error=max(errors),models_sha256=engine.signature,model_artifacts=engine.artifacts,raw_circular_coordinate=12345,raw_source='WT two repeats verified against sample hashes',raw_circular_shape=list(np.asarray(live['views']).shape),review_storage='Append-onlySQLite; restart/read/export verified in test suite',llm='Not integrated; no external data transmission',limitations=['Closed-set classifier cannot establish novel structure types','Probabilities uncalibrated; gradients are sensitivity, not statistical significance','UI uses frozen fold with training/validation overlap notice; exploratory positions remain retrospective']))
print('All344 OOF probabilities reproduced; max error',max(errors),'raw circular probe passed',flush=True)

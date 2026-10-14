"""Compile paired second-round comparisons and group-resampled uncertainty."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from dlpro.experiment_core import save_json
root=Path('results');out=root/'refinement';out.mkdir(exist_ok=True)
d=json.loads((root/'discovery_refined/protocol.json').read_text());metrics=pd.read_csv(root/'discovery_refined/budget_metrics.csv');random=pd.read_csv(root/'discovery_refined/random_trials.csv');candidate=pd.read_csv(root/'discovery_refined/candidates.csv');known=pd.read_csv('data/processed/known_windows.csv');scan=pd.read_csv('data/processed/scan_windows.csv');L=4641652
rng=np.random.default_rng(786);chosen=d['score_choice'];selected=candidate.scan_index.to_numpy();delta=abs(scan.center.to_numpy()[:,None]-known.center.to_numpy()[None]);coverage=np.minimum(delta,L-delta)<=6400
# Independent random selection replay provides per-locus null hit probability for grouped contrasts.
def select(score,budget):
 ids=[];available=np.ones(len(scan),bool)
 for i in np.argsort(-score,kind='stable'):
  if not available[i]:continue
  ids.append(i);distance=abs(scan.center.to_numpy()-scan.center.iloc[i]);available&=np.minimum(distance,L-distance)>=12800
  if len(ids)==budget:return ids
null=np.mean([coverage[select(rng.random(len(scan)),64)].any(0) for _ in range(500)],0);positive=coverage[selected].any(0);test=known.split.eq('test');groups=known.loc[test,'group'].unique();draw=[]
for _ in range(2000):
 ids=np.concatenate([np.flatnonzero(test&known.group.eq(g)) for g in rng.choice(groups,len(groups),replace=True)])
 draw.append((positive[ids]-null[ids]).mean())
summary={'date':'2026-10-14','discovery':d,'discovery_test':metrics[metrics.split.eq('test')].to_dict('records'),'random_test':random[random.split.eq('test')].groupby('budget').center_recall.agg(['mean','std']).reset_index().to_dict('records'),'discovery_group_bootstrap_primary_delta_vs_random_95':np.quantile(draw,[.025,.975]).tolist(),'discovery_test_groups':len(groups)}
sr=pd.read_csv(root/'sr_refined/metrics_summary.csv');summary['sr_native']=sr.groupby(['replicate','method'])[['psnr','ssim']].mean().reset_index().to_dict('records')
summary['sr_native_seed42']=sr[sr.seed.eq(42)].to_dict('records');a=pd.read_csv(root/'sr_refined/projection_ablation_native.csv');summary['sr_projection_native']=a.groupby(['replicate','method'])[['psnr','ssim']].mean().reset_index().to_dict('records')
b=pd.read_csv(root/'sr_refined/known_recovery.csv');summary['sr_shape']=b.groupby('method')[['context_oe_ssim_original','context_oe_ssim_original_100bp']].mean().reset_index().to_dict('records');c=pd.read_csv(root/'sr_refined/projection_ablation_shape.csv');summary['sr_projection_shape']=c.groupby('method')[['oe_ssim_100bp','oe_ssim_200bp']].mean().reset_index().to_dict('records');summary['sr_downstream']=pd.read_csv(root/'sr_refined/downstream_summary.csv').to_dict('records')
paired=c.pivot(index=['structure_id','replicate'],columns='method',values='oe_ssim_100bp');gap=(paired.projected_first_round_cnn-paired.first_round_cnn).groupby('structure_id').mean();ids=gap.index.to_numpy();ci=[gap.loc[rng.choice(ids,len(ids),replace=True)].mean() for _ in range(2000)];summary['sr_projection_100bp_delta_locus_bootstrap_95']=np.quantile(ci,[.025,.975]).tolist()
summary['limitations']=['Retrospective evaluation of previously exposed test positions; no independent external validation','Discovery partially supervised choice among three morphology aggregations on validation labels','200bp conservation imposed by40bp block constraint; not independent learning evidence','New SR training did not outperform projecting existing CNN; preserve negative experiment','Classification recipe unchanged and CHID recall remains34.6%']
save_json(out/'summary.json',summary);print(json.dumps({k:v for k,v in summary.items() if 'bootstrap' in k},indent=2))

"""Conservative whole-context eligibility audit; never relabel exploratory candidates."""
from pathlib import Path
import numpy as np
import pandas as pd
from dlpro.experiment_core import hits,LENGTH,save_json
out=Path('results/candidate_context');out.mkdir(exist_ok=True)
f=pd.read_csv('results/candidate_conditions/unannotated_consensus.csv').merge(pd.read_csv('results/candidate_negative_control/metrics.csv'),on=['candidate_id','center'],validate='one_to_one')
known=pd.read_csv('data/annotations/structures.csv')
f['known_context_overlap']=[bool(hits(int(c),25600,known,buffer=200).any()) for c in f.center]
f['supported_repeat_pass']=(f.wt_oe_correlation_supported>=.5)&(f.supported_triangle_fraction>=.5)
f['eligible_unannotated_context']=f.supported_repeat_pass&~f.known_context_overlap
# Genomic-order greedy independent subset is reproducible, not a maximum-cardinality claim.
selected=[]
for row in f[f.eligible_unannotated_context].sort_values('center').itertuples():
 if all(min(abs(row.center-c),LENGTH-abs(row.center-c))>=25600 for c in selected):selected.append(row.center)
f['independent_context_member']=f.center.isin(selected);f.to_csv(out/'eligibility.csv',index=False)
save_json(out/'protocol.json',dict(core_candidates=len(f),repeat_supported=int(f.supported_repeat_pass.sum()),whole_context_unannotated=int(f.eligible_unannotated_context.sum()),independent_members=int(f.independent_context_member.sum()),context_bp=25600,annotation_buffer_bp=200,minimum_center_separation_bp=25600,selection='Fixed genomic-order greedy subset after QC; no test-driven threshold tuning',limitations=['Consensus candidates exploratory and previously exposed','These are candidate windows, not confirmed new structural classes','No clustering-based novelty claim can be inferred from member counts alone','Mask excludes low observed-contact support, not demonstrated mapping errors']))
print(f[['candidate_id','known_context_overlap','supported_repeat_pass','independent_context_member']].to_string(index=False))

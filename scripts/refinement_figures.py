"""Readable summary figures from saved second-round measurements."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
out=Path('results/refinement');s=json.loads((out/'summary.json').read_text());methods=[s['discovery']['score_choice'],'first_round_AE','density'];labels=['Coherent morphology','First-round AE','Density'];f=pd.DataFrame(s['discovery_test']);r=pd.DataFrame(s['random_test']);fig,axes=plt.subplots(1,2,figsize=(9,3.5),constrained_layout=True)
for ax,budget in zip(axes,[64,128]):
 vals=[float(f[(f.budget==budget)&(f.method==m)].center_recall.iloc[0]) for m in methods];null=float(r[r.budget==budget]['mean'].iloc[0]);ax.bar(range(4),vals+[null],color=['#257F86','#888888','#AA7744','#BBBBBB']);ax.set_xticks(range(4),labels+['Random mean'],rotation=20,ha='right',fontsize=8);ax.set_ylim(0,.7);ax.set_title(f'{budget} candidates');ax.set_ylabel('Held-out known-center recall')
fig.savefig(out/'discovery_budget.png',dpi=150);plt.close(fig)
f=pd.DataFrame(s['sr_projection_shape']);fig,ax=plt.subplots(figsize=(7,3),constrained_layout=True);names=['Bicubic','Original CNN','Projected original CNN','Retrained projected CNN'];values=[next(v['context_oe_ssim_original_100bp'] for v in s['sr_shape'] if v['method']=='bicubic'),float(f[f.method=='first_round_cnn'].oe_ssim_100bp.iloc[0]),float(f[f.method=='projected_first_round_cnn'].oe_ssim_100bp.iloc[0]),next(v['context_oe_ssim_original_100bp'] for v in s['sr_shape'] if v['method']=='subpixel_cnn')];ax.bar(range(4),values,color=['#BBBBBB','#888888','#257F86','#79A5A3']);ax.set_xticks(range(4),names,rotation=10,fontsize=8);ax.set_ylim(.8,1);ax.set_ylabel('100bp off-grid O/E SSIM');fig.savefig(out/'sr_shape.png',dpi=150);plt.close(fig)

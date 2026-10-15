"""Fixed full-arm figures; no example or model chosen by test success."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
out=Path('results/performance_validation');colors=['#4169a1','#c77b30','#388c76'];recipes=['sparse_matched','dense_matched','dense_multiscale'];labels=['64 windows','352 windows','352 + multiscale loss'];fig,axes=plt.subplots(1,3,figsize=(14,4.2),constrained_layout=True)
for recipe,label,color in zip(recipes,labels,colors):
 h=[pd.read_csv(f'results/sr_data_controls/{recipe}_seed{s}_history.csv') for s in [42,43,44]];v=np.stack([a.val_common_objective for a in h]);axes[0].plot(h[0].step,v.mean(0),label=label,color=color);axes[0].fill_between(h[0].step,v.min(0),v.max(0),alpha=.13,color=color)
axes[0].set(xlabel='Optimizer steps',ylabel='Common validation objective',title='Validation only; band = three-seed range');axes[0].legend(fontsize=8)
f=pd.read_csv('results/sr_data_controls/native_metrics.csv');base=f[f.method=='projected_bicubic'].set_index(['sample_index','group','replicate'])
for j,(recipe,label,color) in enumerate(zip(recipes,labels,colors)):
 g=f[f.method.str.startswith(recipe+'_seed')].groupby(['sample_index','group','replicate'])[['psnr']].mean();delta=(g-base[['psnr']]).reset_index()
 for repeat,marker in [(1,'o'),(2,'s')]:
  byblock=delta[delta.replicate==repeat].groupby('group').psnr.mean();x=j+(repeat-1.5)*.18;axes[1].scatter(np.full(len(byblock),x),byblock,s=25,marker=marker,color=color,alpha=.65);axes[1].plot(x,byblock.mean(),'k_',ms=14)
axes[1].set_xticks(range(3),labels,rotation=12,fontsize=8);axes[1].axhline(0,color='black',lw=.7);axes[1].set(ylabel='PSNR delta vs projected bicubic (dB)',title='Eight spatial blocks; circles rep1, squares rep2')
f=pd.read_csv('results/sr_data_controls/shape_metrics.csv');keys=['structure_id','type','group','replicate'];base=f[f.method=='projected_bicubic'].set_index(keys)
for j,(recipe,label,color) in enumerate(zip(recipes,labels,colors)):
 g=f[f.method.str.startswith(recipe+'_seed')].groupby(keys)[['oe_ssim_100bp']].mean();delta=(g-base[['oe_ssim_100bp']]).reset_index();byblock=delta.groupby('group').oe_ssim_100bp.mean();axes[2].scatter(np.full(len(byblock),j),byblock,color=color,alpha=.7);axes[2].plot(j,delta.oe_ssim_100bp.mean(),'k_',ms=14)
axes[2].axhline(0,color='black',lw=.7);axes[2].set_xticks(range(3),labels,rotation=12,fontsize=8);axes[2].set(ylabel='100bp O/E SSIM delta',title='49 loci; points = spatial-block means');fig.suptitle('Fixed SR controls; retrospective spatial evaluation',fontsize=13);fig.savefig(out/'sr_controls.png',dpi=160);plt.close(fig)
if '--sr-only' in sys.argv:raise SystemExit(0)
f=pd.read_csv(out/'classification_comparison.csv').set_index('recipe');names=['fixed_fine_baseline','matched_control_calibrated','repeat_pretrained_calibrated','validation_selected'];fig,ax=plt.subplots(figsize=(10,4.5),constrained_layout=True);x=np.arange(4)
for i,(metric,label) in enumerate([('macro_f1','Macro-F1'),('chid_precision','CHID precision'),('chid_recall','CHID recall'),('chid_f1','CHID F1')]):ax.bar(x+(i-1.5)*.19,f.loc[names,metric],width=.18,label=label)
ax.set_xticks(x,['Fixed baseline','Matched control','Repeat pretraining','Validation-selected']);ax.set(ylim=(0,1),ylabel='OOF score',title='All344 loci,56 spatial groups; retrospective comparison');ax.legend(ncol=4,fontsize=8);fig.savefig(out/'classification_controls.png',dpi=160);plt.close(fig)

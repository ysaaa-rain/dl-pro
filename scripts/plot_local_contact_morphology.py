"""Render local-geometry validation and matched known-reference comparisons."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=Path('results/local_contact_morphology')
known=pd.read_csv('data/processed/known_windows.csv');candidates=pd.read_csv(p/'localized_candidates.csv');z=np.load(p/'measurements.npz')
for row in candidates.itertuples():
 i=len(known)+row.scan_index;idx=int(known.index[known.structure_id==row.nearest_known_shape_id][0])
 fig,axes=plt.subplots(3,4,figsize=(13,9.5),constrained_layout=True)
 for channel,bp in enumerate([50,100,200]):
  for ax,(source,r,title) in zip(axes[channel],[(i,1,f'{row.candidate_id} rep1'),(i,2,f'{row.candidate_id} rep2'),(idx,1,f'{row.nearest_known_shape_id} rep1'),(idx,2,f'{row.nearest_known_shape_id} rep2')]):
   im=ax.imshow(np.ma.masked_where(~z[f'masks_rep{r}'][source,channel],z[f'images_rep{r}'][source,channel]),origin='lower',cmap='magma',vmin=0,vmax=2)
   ax.set(title=f'{title}, {bp} bp',xlabel='Aligned footprint bin')
 fig.colorbar(im,ax=axes,label='Amplitude-normalized log(1+O/E)',shrink=.7)
 fig.savefig(p/row.comparison_heatmap_file,dpi=130);plt.close(fig)
summary=pd.read_csv(p/'known_classification_summary.csv');local=pd.read_csv(p/'known_localization_summary.csv')
fig,axes=plt.subplots(1,3,figsize=(15,4.5),constrained_layout=True)
methods=['fixed_geometry','localized_geometry','aligned_image'];x=np.arange(3)
for r,offset in [(1,-.17),(2,.17)]:
 vals=summary[summary.repeat==r].set_index('method').loc[methods,'macro_f1'].to_numpy()
 axes[0].bar(x+offset,vals,width=.34,label=f'WT rep{r}')
axes[0].set_xticks(x);axes[0].set_xticklabels(['Whole-window\ngeometry','Localized\ngeometry','Aligned\nimage']);axes[0].set(ylabel='Spatial OOF Macro-F1',ylim=(0,.8));axes[0].legend()
for kind in ['CHIN','OPCID','CHID']:
 sub=local[local.actual_type==kind]
 axes[1].plot(sub.bin_bp,sub.rep1_annotated_interval_iou,marker='o',label=kind)
axes[1].set(xlabel='Bin size (bp)',ylabel='Median estimated/annotated interval IoU',ylim=(0,1));axes[1].legend()
axes[2].barh(candidates.candidate_id,candidates.repeat_boundary_iou,color=np.where(candidates.repeat_boundary_iou>=.5,'steelblue','darkorange'))
axes[2].set(xlabel='Independent repeat boundary IoU',xlim=(0,1.05));axes[2].axvline(.5,color='gray',ls='--',lw=1)
fig.savefig(p/'geometry_validation.png',dpi=150);plt.close(fig)

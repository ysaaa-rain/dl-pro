"""Readable three-panel structure restoration plots, generated from model outputs."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from dlpro.experiment_core import ContactReader,ShapeTransform,SubpixelSR,centered_indices,configure
from dlpro.superresolution import mean_pool

out=Path('results/report_figures');out.mkdir(parents=True,exist_ok=True);configure(42)
protocol=pd.read_json('results/sr_expanded/protocol.json',typ='series');scale=float(protocol.normalization_scale)
model=SubpixelSR();model.load_state_dict(torch.load('results/sr_expanded/model_seed42.pt',weights_only=True)['state_dict']);model.eval()
samples=pd.read_csv('data/annotations/samples.csv');reader=ContactReader(samples[(samples.condition=='37C')&(samples.replicate==1)].path.iloc[0])
known=pd.read_csv('results/sr_expanded/known_test_windows.csv');folds=pd.read_csv('results/classification_cv/out_of_fold_predictions.csv').set_index('structure_id')
for label in ['CHID','CHIN','OPCID']:
    row=known[known.type==label].iloc[0];fold=int(folds.loc[row.structure_id].fold)
    tf=ShapeTransform.load('results/classification/transform.npz' if fold==0 else f'results/classification_cv/transform_fold{fold}.npz')
    start=int(row.window_start);counts=reader.window(start//10,n=2560,bin_bp=10,depth_normalize=False)
    low=np.clip(np.log1p(mean_pool(counts))/scale,0,1).astype('float32')
    with torch.no_grad():
        linear=F.interpolate(torch.from_numpy(low[None,None]),scale_factor=4,mode='bicubic',align_corners=False).clamp(0,1).numpy()[0,0]
        restored=model(torch.from_numpy(low[None,None])).clamp(0,1).numpy()[0,0]
    small=(centered_indices(int(row.center))[0]*100-start)//10
    shapes=[]
    for a in [counts,np.expm1(linear*scale),np.expm1(restored*scale)]:
        features=np.stack([mean_pool(a[small:small+1280,small:small+1280],10),mean_pool(a,20)])*1e6/reader.library_size
        shapes.append(tf.transform(features[None])[0,1])
    np.savez_compressed(out/f'{label}_shape_comparison.npz',original=shapes[0],bicubic=shapes[1],cnn=shapes[2])
    fig,axes=plt.subplots(1,3,figsize=(9,3.4),constrained_layout=True)
    for ax,a,title in zip(axes,shapes,['Original','Bicubic','CNN']):
        im=ax.imshow(a,vmin=0,vmax=1,cmap='magma',origin='lower');aa=(row.start-start)/200;bb=(row.end-start)/200
        ax.add_patch(Rectangle((aa,aa),bb-aa,bb-aa,fill=False,edgecolor='cyan',lw=1));ax.set_title(title,fontsize=12);ax.set_xlabel('200 bp bin',fontsize=10);ax.tick_params(labelsize=9)
    axes[0].set_ylabel('200 bp bin',fontsize=10);fig.colorbar(im,ax=axes,label='Scaled log(1+O/E)',shrink=.8);fig.suptitle(row.structure_id,fontsize=12)
    fig.savefig(out/f'{label}_restoration.png',dpi=180);plt.close(fig)

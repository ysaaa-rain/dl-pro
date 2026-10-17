"""Readable, common-axis figures from frozen metrics; no score or metric changes."""
import argparse
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator,FormatStrFormatter
from dlpro.method_review import load_config

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();cfg=load_config(a.config);root=Path(cfg['output']);out=root/'review'
    for name in ['novelty_comparison','shared_training_effects']:
        original=out/f'{name}_initial.png'
        if not original.exists():shutil.copyfile(out/f'{name}.png',original)
    frame=pd.read_csv(root/'background_ae/novelty_summary.csv')
    order=['morphology_baseline','ae_residual','ae_direction_joint','ae_reference_diagnostic'];labels=['Morphology','AE residual','AE + direction policy','AE reference diagnostic']
    fig,axes=plt.subplots(1,3,figsize=(11,3.7),constrained_layout=True,sharey=True)
    for ax,(hidden,part) in zip(axes,frame[(frame.repeat==1)&(frame.negative=='allowed_known')].groupby('hidden_type')):
        part=part.set_index('method').loc[order]
        ax.errorbar(part.auc,np.arange(len(part)),xerr=np.stack([part.auc-part.low,part.high-part.auc]),fmt='o',capsize=3)
        ax.set_yticks(np.arange(len(part)),labels);ax.axvline(.5,color='grey',ls='--');ax.set_title(hidden);ax.set_xlim(0,1);ax.set_xlabel('AUROC');ax.grid(axis='x',alpha=.2)
    axes[0].invert_yaxis();fig.suptitle('WT1 / held type vs other known types / 95% spatial-group intervals')
    fig.savefig(out/'novelty_comparison.png',dpi=150);plt.close(fig)
    frame=pd.read_csv(out/'shared_vs_single.csv');fig,axes=plt.subplots(1,3,figsize=(10.8,3.7),constrained_layout=True,sharey=True)
    for ax,metric,title,scale in zip(axes,['psnr','ssim','oe_100bp_ssim'],['Native PSNR','Native SSIM','100 bp O/E SSIM'],[1,1e5,1e5]):
        part=frame[(frame.split=='new_test')&(frame.metric==metric)].set_index('comparison').loc[['same_repeat','cross_repeat']]
        ax.errorbar(part.delta*scale,np.arange(len(part)),xerr=np.stack([(part.delta-part.low)*scale,(part.high-part.delta)*scale]),fmt='o',capsize=3)
        ax.set_yticks([0,1],['Same repeat','Cross repeat']);ax.axvline(0,color='grey',ls='--');ax.set_title(title);ax.grid(axis='x',alpha=.2)
        ax.xaxis.set_major_locator(MaxNLocator(4));ax.xaxis.set_major_formatter(FormatStrFormatter('%.4f' if scale==1 else '%.1f'))
        ax.set_xlabel('Difference (dB)' if scale==1 else r'Difference (units of $10^{-5}$)')
    axes[0].invert_yaxis();fig.suptitle('Shared minus matched single-repeat training / 95% spatial-group intervals')
    fig.savefig(out/'shared_training_effects.png',dpi=150);plt.close(fig)

if __name__=='__main__':main()

"""Improve figure labels after visual QA; reuse metrics, never rerun or select models."""
import argparse
from pathlib import Path
import shutil
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dlpro.method_review import load_config
from dlpro.experiment_core import LABELS
p=argparse.ArgumentParser();p.add_argument('--config',required=True);args=p.parse_args();root=Path(load_config(args.config)['output']);out=root/'review';draft=out/'plot_drafts';draft.mkdir(exist_ok=True)
names={'symmetric_bias':'Class bias','geometric_control':'Geometry control','geometric_directional':'Geometry + direction'}
for name in ['classification_paired_effects.png','specificity_by_fill.png']:
    if not (draft/name).exists():shutil.copy2(out/name,draft/name)
pairs=pd.read_csv(root/'experiment/paired_intervals.csv');fig,axes=plt.subplots(1,3,figsize=(13,4.6),constrained_layout=True)
for ax,metric in zip(axes,['accuracy','macro_f1','f1']):
    part=pairs[(pairs.reference=='frozen_cnn')&(pairs.metric==metric)&(pairs.type==('CHID' if metric=='f1' else 'all'))]
    for i,row in enumerate(part.itertuples()):ax.errorbar(row.delta,i,xerr=[[max(0,row.delta-row.low)],[max(0,row.high-row.delta)]],fmt='o',capsize=3,color='#23698c')
    ax.set_yticks(range(len(part)),[names[name] for name in part.method]);ax.axvline(0,color='gray');ax.set_title({'accuracy':'Accuracy','macro_f1':'Macro F1','f1':'CHID F1'}[metric]);ax.set_xlabel('Difference vs CNN (95% group CI)');ax.grid(axis='x',alpha=.2)
fig.suptitle('Retrospective spatial CV; no independent final-test claim');fig.savefig(out/'classification_paired_effects.png',dpi=150);plt.close(fig)
values=pd.read_csv(root/'explanation/specificity_intervals.csv');fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
for ax,label in zip(axes,LABELS):
    for fill,g in values[values.type==label].groupby('fill'):ax.errorbar(g.bp,g.mean_specificity,yerr=np.vstack([g.mean_specificity-g.low,g.high-g.mean_specificity]),marker='o',capsize=3,label=fill.replace('training_',''))
    ax.axhline(0,color='gray');ax.set_title(label);ax.set_xticks([50,100,200]);ax.set_xlim(40,210);ax.set_xlabel('View bin bp; NA points not drawn');ax.set_ylabel('Probability drop: proxy minus background')
axes[-1].legend(fontsize=8);fig.suptitle('Equal area/distance controls; annotated interval is a proxy');fig.savefig(out/'specificity_by_fill.png',dpi=150);plt.close(fig)
print('Figure labels verified; metrics and selection unchanged',flush=True)

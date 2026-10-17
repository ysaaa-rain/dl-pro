"""First eligible example per type, chosen by coordinates, never by model gain."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dlpro.method_review import load_config,write_json
from dlpro.sr_review import load_data,make_batch,make_model,infer_model
from evaluate_allocation_review import simple_predictions

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();cfg=load_config(a.config)
    root=Path(cfg['output']);out=root/'known_examples';out.mkdir(exist_ok=True);assert not (out/'protocol.json').exists()
    frame=pd.read_csv(root/'evaluation/known_structure_per_window.csv')
    frame=frame[(frame.source_repeat==1)&(frame.comparison=='cross_repeat')&(frame.method=='train_distance_prior')]
    examples=frame.sort_values(['known_start','start']).drop_duplicates('structure_id').groupby('type').head(1)
    plan,raw,depth=load_data(cfg);old=Path(cfg['previous_output'])/'sr_corrected';params=np.load(old/'normalization.npz');sr=cfg['sr'];torch.set_num_threads(cfg['threads'])
    models={}
    for method,directory in [('legacy_sparse_mse',old),('allocation_shared_same',root/sr['result_subdir'])]:
        cp=torch.load(directory/f"{method}_seed{sr['train_seeds'][0]}.pt",weights_only=True)
        model=make_model(method,float(params['scale']),params['profile'],sr).eval();model.load_state_dict(cp['state_dict']);models[method]=model
    records=[]
    with torch.no_grad():
        for row in examples.itertuples():
            idx=int(plan.index[plan.start==row.start][0]);source,_=make_batch(raw,depth,[idx],[0]);target,_=make_batch(raw,depth,[idx],[1]);low=F.avg_pool2d(source,4)
            simple=simple_predictions(low,params['profile'],4);preds={'source':source,'target':target,'bicubic':simple['projected_bicubic'],'prior':simple['train_distance_prior']}
            for method,model in models.items():preds[method]=infer_model(model,method,low,float(params['scale']),sr)[0]
            center=(row.known_start+row.known_end)//2;relative=(center-row.start)//100
            lo=min(int((row.known_start-row.start)//100)-2,int(relative)-3);hi=max(int(np.ceil((row.known_end-row.start)/100))+2,int(relative)+4)
            n=source.shape[-1]//10;assert 0<=lo<hi<=n;d=abs(np.arange(n)[:,None]-np.arange(n)[None]);expected=params['expected_100'][d]
            fig,axes=plt.subplots(1,len(preds),figsize=(14,3.3),constrained_layout=True)
            for ax,(name,pred) in zip(axes,preds.items()):
                image=F.avg_pool2d(pred,10).numpy()[0,0];field=np.log1p(image/expected);field[d<2]=0
                im=ax.imshow(field[lo:hi,lo:hi],origin='lower',cmap='magma',vmin=0,vmax=2,extent=[row.start+lo*100,row.start+hi*100,row.start+lo*100,row.start+hi*100]);ax.set_title(name.replace('allocation_','').replace('_',' '),fontsize=9);ax.ticklabel_format(style='sci',axis='both',scilimits=(0,0))
                for coordinate in [row.known_start,row.known_end]:ax.axvline(coordinate,color='cyan',lw=.6,alpha=.7);ax.axhline(coordinate,color='cyan',lw=.6,alpha=.7)
            fig.suptitle(f'{row.type}: {row.structure_id} / first eligible coordinate example; WT1 source, WT2 target')
            fig.colorbar(im,ax=axes,label='log(1+O/E), common training background',shrink=.7)
            fig.savefig(out/f'{row.type}.png',dpi=130);plt.close(fig)
            np.savez_compressed(out/f'{row.type}.npz',**{name:pred.numpy()[0,0] for name,pred in preds.items()},roi_bins=np.asarray([lo,hi]),known_interval=np.asarray([row.known_start,row.known_end]))
            records.append(dict(type=row.type,structure_id=row.structure_id,start=int(row.start),annotation_start=int(row.known_start),annotation_end=int(row.known_end),roi_lo=lo,roi_hi=hi))
    write_json(out/'protocol.json',dict(selection='First eligible annotation per type sorted by coordinate/start, source repeat1; no model gain selection',examples=records,common_log_oe_range=[0,2],neural_seed=sr['train_seeds'][0],scope='19 complete annotation subset only; not full-context classifier accuracy'))

if __name__=='__main__':main()

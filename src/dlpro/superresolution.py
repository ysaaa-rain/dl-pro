"""Small real-data SR pilot, native 10 bp target, 40 bp input.

Not the final whole-genome experiment. Regions are sampled without annotation
selection. Split by genomic blocks before expansion into biological repeats.
"""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from skimage.metrics import structural_similarity
from .data import read_window


def block_plan(length, seed=42, patch_bins=128, bin_size=10, block_bp=100000):
    if patch_bins <= 0 or patch_bins % 4 or block_bp < 4*patch_bins*bin_size:
        raise ValueError('Invalid pilot geometry')
    # Only full blocks, avoid both the circular seam and chromosome tail.
    blocks = np.arange(length // block_bp)
    if len(blocks) < 10: raise ValueError('Need at least ten blocks')
    order = np.random.default_rng(seed).permutation(blocks)
    n_train, n_val = int(.7*len(blocks)), int(.15*len(blocks))
    splits = {int(b): ('train' if i<n_train else 'val' if i<n_train+n_val else 'test')
              for i,b in enumerate(order)}
    rows = []
    for b in blocks:
        for offset in [.25,.75]:
            start = int((b*block_bp + offset*block_bp)//bin_size)*bin_size
            rows.append(dict(group=int(b), start=start, end=start+patch_bins*bin_size, split=splits[int(b)]))
    return pd.DataFrame(rows)


def mean_pool(a, factor=4):
    if a.shape[-1] != a.shape[-2] or a.shape[-1] % factor:
        raise ValueError('Square dimensions must be divisible by factor')
    n = a.shape[-1]//factor
    return a.reshape(*a.shape[:-2], n,factor,n,factor).mean(axis=(-3,-1))


class ResidualSR(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(1,16,3,padding=1),nn.ReLU(),
            nn.Conv2d(16,16,3,padding=1),nn.ReLU(),nn.Conv2d(16,1,3,padding=1))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
    def forward(self, low):
        base = F.interpolate(low, scale_factor=4, mode='bilinear', align_corners=False)
        out = base + self.net(base)
        return (out + out.transpose(-1,-2))/2


def scores(target, pred):
    pred = np.clip(pred,0,1)
    mse = float(np.mean((target-pred)**2))
    return dict(mse=mse, psnr=float(-10*np.log10(max(mse,1e-12))),
        ssim=float(structural_similarity(target,pred,data_range=1)),
        off_diagonal_mse=float(np.mean((target-pred)[~np.eye(len(target),dtype=bool)]**2)))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--raw', default='data/raw')
    p.add_argument('--output', default='results/sr_pilot')
    p.add_argument('--dataset', help='Reuse patches.npz and windows.csv from a previous run')
    p.add_argument('--epochs', type=int, default=12)
    p.add_argument('--seed', type=int, default=42)
    a=p.parse_args()
    if a.epochs < 1: raise ValueError('epochs must be positive')
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    torch.use_deterministic_algorithms(True)
    if a.dataset:
        source=Path(a.dataset)
        frame=pd.read_csv(source/'windows.csv')
        raw=np.load(source/'patches.npz')['raw']
        expected=block_plan(4641652,a.seed)
        for replicate,part in frame.groupby('replicate'):
            if not part[['group','start','end','split']].reset_index(drop=True).equals(expected):
                raise ValueError('Cached split does not match seed and pilot geometry')
        if len(raw)!=len(frame) or raw.shape[1:]!=(128,128) or frame.replicate.nunique()!=2:
            raise ValueError('Invalid cached dataset')
        plan=expected
    else:
        paths=sorted(Path(a.raw).glob('GSE272159_37C_rep*.mapq_30.10.cool'))
        if len(paths)!=2: raise ValueError('Expected precisely the two WT replicates')
        import cooler
        metadata=[cooler.Cooler(str(path)) for path in paths]
        if any(c.binsize != 10 or c.chromsizes.to_dict() != {'NC_000913.3':4641652} for c in metadata):
            raise ValueError('Unexpected genome or bin size')
        plan=block_plan(4641652,a.seed)
        windows=[]; counts=[]
        for repeat,path in enumerate(paths,1):
            for i,r in plan.iterrows():
                matrix,mask=read_window(str(path),'NC_000913.3',int(r.start),int(r.end),max_bins=128)
                if not mask.all(): raise ValueError('Pilot requires finite raw counts')
                counts.append(matrix.astype('float32'))
                windows.append(dict(r,replicate=repeat,chrom='NC_000913.3',source=str(path)))
                if i%20==0: print(f'extracted repeat {repeat}, window {i+1}/{len(plan)}',flush=True)
        frame=pd.DataFrame(windows)
        raw=np.stack(counts)
    frame.to_csv(out/'windows.csv',index=False)
    # Block averaging of the symmetric count image precedes log transformation.
    # This is image-resolution degradation, not read-depth subsampling.
    low=mean_pool(raw)
    train=frame.split.eq('train').to_numpy(); val=frame.split.eq('val').to_numpy()
    test=frame.split.eq('test').to_numpy()
    scale=float(np.quantile(np.log1p(raw[train]),.995))
    if scale<=0: raise ValueError('Degenerate training intensity')
    high=np.clip(np.log1p(raw)/scale,0,1).astype('float32')
    low=np.clip(np.log1p(low)/scale,0,1).astype('float32')
    np.savez_compressed(out/'patches.npz',raw=raw,low=low,target=high)
    x=torch.from_numpy(low[:,None]); y=torch.from_numpy(high[:,None])
    model=ResidualSR(); optim=torch.optim.Adam(model.parameters(),lr=1e-3)
    rng=np.random.default_rng(a.seed); best=float('inf'); state=None; history=[]
    for epoch in range(a.epochs):
        model.train(); losses=[]
        for ids in np.array_split(rng.permutation(np.flatnonzero(train)),max(1,int(train.sum()/16))):
            optim.zero_grad(); loss=F.mse_loss(model(x[ids]),y[ids]);loss.backward();optim.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad(): vl=float(F.mse_loss(model(x[val]).clamp(0,1),y[val]))
        history.append(dict(epoch=epoch+1,train_mse=float(np.mean(losses)),val_mse=vl))
        if vl<best: best=vl; state=copy.deepcopy(model.state_dict())
        print(json.dumps(history[-1]),flush=True)
    model.load_state_dict(state); model.eval()
    torch.save(dict(state_dict=state,scale=scale,seed=a.seed,architecture='ResidualSR',factor=4),out/'model.pt')
    rows=[]
    with torch.no_grad():
        prediction=model(x[test]).clamp(0,1).numpy()[:,0]
        baseline={mode:F.interpolate(x[test],scale_factor=4,mode=mode,align_corners=False).clamp(0,1).numpy()[:,0]
                  for mode in ['bilinear','bicubic']}
    preds=dict(baseline,residual_cnn=prediction)
    for local,idx in enumerate(np.flatnonzero(test)):
        for name,values in preds.items():
            rows.append(dict(sample_index=int(idx),replicate=int(frame.iloc[idx].replicate),method=name,
                             **scores(high[idx],values[local])))
    metrics=pd.DataFrame(rows);metrics.to_csv(out/'metrics_per_window.csv',index=False)
    summary=metrics.groupby(['replicate','method'])[['mse','psnr','ssim','off_diagonal_mse']].mean()
    summary.to_csv(out/'metrics_summary.csv')
    pd.DataFrame(history).to_csv(out/'history.csv',index=False)
    np.savez_compressed(out/'test_predictions.npz',sample_indices=np.flatnonzero(test),**preds)
    protocol=dict(seed=a.seed,epochs=a.epochs,selected_epoch=min(history,key=lambda r:r['val_mse'])['epoch'],
        target_bin_bp=10,input_bin_bp=40,patch_bins=128,window_bp=1280,scale=scale,
        n_by_split=frame.split.value_counts().to_dict(),
        blocks_by_split=plan.groupby('split').group.nunique().to_dict(),
        normalization='log1p / training-only pooled .995 quantile; clip 0..1',
        limitations=['pilot windows cover only a sparse subset of genome','raw unbalanced contacts',
            '1280 bp windows do not cover most complete annotated structures',
            'both repeats participate in training on train blocks; this is spatial holdout, not repeat-disjoint training',
            'metrics on transformed intensity, not proof of biological structure recovery',
            'single seed pilot; no hyperparameter search'],
        python=__import__('sys').version,torch=torch.__version__,numpy=np.__version__)
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ids=np.flatnonzero(test)
    fig,axes=plt.subplots(2,4,figsize=(12,6),constrained_layout=True)
    for r,local in enumerate([0,len(ids)//2]):
        values=[high[ids[local]],baseline['bilinear'][local],baseline['bicubic'][local],prediction[local]]
        for ax,value,title in zip(axes[r],values,['Target 10 bp','Bilinear','Bicubic','Residual CNN']):
            im=ax.imshow(value,vmin=0,vmax=1,cmap='magma',origin='lower')
            ax.set_title(title);ax.set_xlabel('10 bp bin')
        axes[r,0].set_ylabel(f"WT rep{frame.iloc[ids[local]].replicate}\n10 bp bin")
    fig.colorbar(im,ax=axes,label='Normalized log count',shrink=.8)
    fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
    print(summary.to_string(),flush=True)

if __name__=='__main__':main()

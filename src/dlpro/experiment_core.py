"""Circular, sparse Micro-C windows and train-only transformations."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import cooler
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.model_selection import StratifiedGroupKFold
from .data import overlap_groups

LENGTH=4641652
CHROM='NC_000913.3'
LABELS=['CHID','CHIN','OPCID']


def save_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


class ContactReader:
    def __init__(self,path):
        self.c=cooler.Cooler(str(path));self.path=str(path)
        if self.c.binsize!=10 or self.c.chromsizes.to_dict()!={CHROM:LENGTH}:
            raise ValueError('Unexpected Micro-C axes')
        self.n_native=int(self.c.info['nbins']);self.library_size=int(self.c.info['sum'])
        self.matrix=self.c.matrix(balance=False,sparse=True)

    def window(self,start,n=128,bin_bp=100,depth_normalize=True):
        """Circular coarse-bin window, averaged per 10bp-by-10bp contact pixel.

        Sparse image aggregation (not unique-pair aggregation). The 52bp final
        coarse bin is divided by its actual area instead of a full 100bp bin.
        """
        if bin_bp%10 or n<=0:raise ValueError('Invalid bin geometry')
        factor=bin_bp//10;total=(LENGTH+bin_bp-1)//bin_bp
        if n>total:raise ValueError('Window larger than chromosome')
        first=int(start)%total;ids=(first+np.arange(n))%total
        count=min(n,total-first)
        segments=[(first,count,0)]
        if count<n:segments.append((0,n-count,count))
        result=np.zeros((n,n),dtype=np.float64)
        for rs,rn,ro in segments:
            for cs,cn,co in segments:
                pixels=self.matrix[rs*factor:min((rs+rn)*factor,self.n_native),cs*factor:min((cs+cn)*factor,self.n_native)].tocoo()
                flat=(pixels.row//factor+ro)*n+(pixels.col//factor+co)
                result+=np.bincount(flat,weights=pixels.data,minlength=n*n).reshape(n,n)
        widths=np.minimum(bin_bp,LENGTH-ids*bin_bp).astype(float)
        result/=np.outer(widths/10,widths/10)
        if depth_normalize:result*=1e6/self.library_size
        if not np.isfinite(result).all() or not np.allclose(result,result.T) or (result<0).any():
            raise ValueError('Invalid extracted matrix')
        return result.astype(np.float32)


def physical_intervals(center,width):
    start=(int(center)-width//2)%LENGTH;end=start+width
    return [(start,min(end,LENGTH))]+([(0,end-LENGTH)] if end>LENGTH else [])


def hits(center,width,annotations,buffer=0):
    result=np.zeros(len(annotations),bool)
    for a,b in physical_intervals(center,width):
        result|=(annotations.start.to_numpy()<b+buffer)&(annotations.end.to_numpy()>a-buffer)
    return result


def centered_indices(center):
    return [int(np.floor((center+50)/100))-64,int(np.floor((center+100)/200))-64]


def plan_known():
    f=pd.read_csv('data/annotations/structures.csv')
    starts=[(centered_indices(int(x))[1])*200 for x in f.center]
    ext=pd.DataFrame(dict(chrom=f.chrom,start=starts,end=np.asarray(starts)+25600))
    if (ext.start<0).any() or (ext.end>LENGTH).any():
        raise ValueError('Class-split circular seam grouping needs explicit extension')
    f['window_start']=ext.start;f['window_end']=ext.end;f['group']=overlap_groups(ext)
    splitter=StratifiedGroupKFold(5,shuffle=True,random_state=42)
    fold=np.zeros(len(f),int)
    for i,(_,ix) in enumerate(splitter.split(f,f.type,f.group)):fold[ix]=i
    f['split']=np.where(fold==0,'test',np.where(fold==1,'val','train'))
    for split in ['train','val','test']:
        if set(f[f.split==split].type)!=set(LABELS):raise ValueError('Missing class in split')
    # Prove that actual maximum input extents do not cross split boundaries.
    for i,row in ext.iterrows():
        overlap=(ext.start<row.end)&(ext.end>row.start)
        if f.loc[overlap,'split'].nunique()!=1:raise ValueError('Window overlap leakage')
    f.to_csv('data/processed/known_windows.csv',index=False)
    save_json('results/experiment/split_audit.json',dict(
        seed=42,groups=int(f.group.nunique()),counts=pd.crosstab(f.split,f.type).to_dict(),
        input_widths_bp=[12800,25600],max_window_groups_disjoint=True,
        repeats_share_split=True,full_structure_coverage=bool((f.start>=f.window_start).all() and (f.end<=f.window_end).all())))
    return f


class ShapeTransform:
    def fit(self,raw):
        if raw.ndim!=4 or len(raw)==0:raise ValueError('Expected NxCxHxW training tensor')
        self.expected=np.stack([np.stack([np.diagonal(raw[:,c],offset=d,axis1=1,axis2=2).mean(axis=1) for d in range(raw.shape[-1])]).mean(axis=1) for c in range(raw.shape[1])])
        self.expected=np.maximum(self.expected,1e-10)
        transformed=self.unscaled(raw)
        self.scale=np.maximum(np.quantile(transformed,.995,axis=(0,2,3)),1e-6).astype(np.float32)
        return self
    def unscaled(self,raw):
        n=raw.shape[-1];distance=abs(np.arange(n)[:,None]-np.arange(n)[None,:])
        values=np.log1p(raw/self.expected[:,distance][None])
        values[:,:,distance<2]=0
        return values
    def transform(self,raw):
        return np.clip(self.unscaled(raw)/self.scale[None,:,None,None],0,1).astype(np.float32)
    def save(self,path):np.savez(path,expected=self.expected,scale=self.scale,mode=type(self).__name__)
    @classmethod
    def load(cls,path):
        a=np.load(path);constructor=DensityInvariantTransform if 'mode' in a.files and str(a['mode'])=='DensityInvariantTransform' else cls
        obj=constructor();obj.expected=a['expected'];obj.scale=a['scale'];return obj


class DensityInvariantTransform(ShapeTransform):
    """Cancel overall contact amplitude before shape encoding."""
    def unscaled(self,raw):
        n=raw.shape[-1];distance=abs(np.arange(n)[:,None]-np.arange(n)[None,:])
        valid=distance>=2
        oe=raw/self.expected[:,distance][None]
        amplitude=oe[:,:,valid].mean(axis=-1)
        amplitude=np.where(amplitude>0,amplitude,1)
        values=np.log1p(oe/amplitude[:,:,None,None])
        values[:,:,~valid]=0
        return values


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv=nn.Sequential(nn.Conv2d(2,16,5,stride=2,padding=2),nn.ReLU(),nn.Conv2d(16,32,3,stride=2,padding=1),nn.ReLU(),nn.Conv2d(32,48,3,stride=2,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d((4,4)))
        self.embedding=nn.Sequential(nn.Flatten(),nn.Linear(48*16,64),nn.ReLU())
        self.head=nn.Linear(64,3)
    def encode(self,x):return self.embedding(self.conv(x))
    def forward(self,x):return self.head(self.encode(x))


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(nn.Conv2d(1,8,4,2,1),nn.ReLU(),nn.Conv2d(8,16,4,2,1),nn.ReLU(),nn.Conv2d(16,16,4,2,1),nn.ReLU(),nn.Conv2d(16,16,4,2,1),nn.ReLU(),nn.Flatten(),nn.Linear(16*8*8,32))
        self.expand=nn.Sequential(nn.Linear(32,16*8*8),nn.ReLU())
        self.decoder=nn.Sequential(nn.ConvTranspose2d(16,16,4,2,1),nn.ReLU(),nn.ConvTranspose2d(16,16,4,2,1),nn.ReLU(),nn.ConvTranspose2d(16,8,4,2,1),nn.ReLU(),nn.ConvTranspose2d(8,1,4,2,1),nn.Sigmoid())
    def forward(self,x):return self.decoder(self.expand(self.encoder(x)).reshape(-1,16,8,8))


def configure(seed):
    torch.set_num_threads(4);torch.manual_seed(seed);np.random.seed(seed)
    torch.use_deterministic_algorithms(True)


class SubpixelSR(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Conv2d(1,8,3,padding=1),nn.ReLU(),nn.Conv2d(8,8,3,padding=1),nn.ReLU(),nn.Conv2d(8,16,3,padding=1),nn.PixelShuffle(4))
        nn.init.zeros_(self.net[4].weight);nn.init.zeros_(self.net[4].bias)
    def forward(self,low):
        base=F.interpolate(low,scale_factor=4,mode='bilinear',align_corners=False)
        result=base+self.net(low);return (result+result.transpose(-1,-2))/2

"""Second-round spatial scoring and count-consistent reconstruction primitives."""
import numpy as np
from scipy.ndimage import gaussian_filter
import torch
from torch import nn
from torch.nn import functional as F
from dlpro.experiment_core import SubpixelSR


def morphology_features(raw, expected, valid_rows=None):
    """Positive coherent enrichment; mask zero rows and near diagonals before smoothing."""
    n=raw.shape[-1];d=abs(np.arange(n)[:,None]-np.arange(n)[None,:]);rows=raw.sum(-1)>0
    if valid_rows is not None:
        if valid_rows.shape!=rows.shape:raise ValueError("Support mask shape mismatch")
        rows=rows&valid_rows
    valid=(d[None,None]>=2)&rows[:,:,:,None]&rows[:,:,None,:]
    oe=raw/expected[:,d][None];amplitude=(oe*valid).sum((-2,-1))/np.maximum(valid.sum((-2,-1)),1)
    z=np.log1p(oe/np.maximum(amplitude[:,:,None,None],1e-12));z=np.where(valid,z,0)
    center=np.zeros((n,n),bool);center[n//4:3*n//4,n//4:3*n//4]=True
    result=[]
    for c in range(raw.shape[1]):
        mask=valid[:,c].astype(float);field=z[:,c];eligible=valid[:,c]&center
        def smooth(sigma):return gaussian_filter(field,(0,sigma,sigma))/np.maximum(gaussian_filter(mask,(0,sigma,sigma)),1e-6)
        broad=smooth(12)
        for sigma in [1,2,4,8]:
            contrast=np.maximum(smooth(sigma)-broad,0)
            result.append(np.asarray([np.quantile(v[m],.98) if m.any() else 0 for v,m in zip(contrast,eligible)]))
    return np.stack(result,axis=1)


def project_counts(counts, low_counts):
    """Each 4x4 high-resolution block has exactly the measured low-resolution mean."""
    weights=counts.clamp_min(1e-8)
    weights=(weights+weights.transpose(-1,-2))/2
    average=F.avg_pool2d(weights,4)
    return weights/F.interpolate(average,scale_factor=4,mode='nearest')*F.interpolate(low_counts,scale_factor=4,mode='nearest')


class CountConsistentSR(SubpixelSR):
    def __init__(self,scale):super().__init__();self.scale=float(scale)
    def forward(self,low):
        low_counts=torch.expm1(low*self.scale)
        prior=F.interpolate(low_counts,scale_factor=4,mode='bicubic',align_corners=False).clamp_min(1e-5)
        correction=self.net(low);correction=(correction+correction.transpose(-1,-2))/2
        counts=project_counts(prior*torch.exp(correction.clamp(-4,4)),low_counts)
        return torch.log1p(counts)/self.scale


class DilatedResidual(nn.Module):
    def __init__(self,channels,dilation):
        super().__init__();self.layers=nn.Sequential(nn.Conv2d(channels,channels,3,padding=dilation,dilation=dilation),nn.ReLU(),nn.Conv2d(channels,channels,3,padding=1))
    def forward(self,x):return x+.1*self.layers(x)


class ContextCountSR(nn.Module):
    def __init__(self,scale,channels=16,dilations=(1,2,4,8,16,32)):
        super().__init__();self.scale=float(scale);self.encoder=nn.Sequential(nn.Conv2d(1,channels,3,padding=1),nn.ReLU(),*[DilatedResidual(channels,d) for d in dilations])
        self.output=nn.Sequential(nn.Conv2d(channels,16,3,padding=1),nn.PixelShuffle(4));nn.init.zeros_(self.output[0].weight);nn.init.zeros_(self.output[0].bias)
    def forward(self,low):
        counts=torch.expm1(low*self.scale);prior=F.interpolate(counts,scale_factor=4,mode='bicubic',align_corners=False).clamp_min(1e-5);correction=self.output(self.encoder(low));correction=(correction+correction.transpose(-1,-2))/2
        result=project_counts(prior*torch.exp(correction.clamp(-4,4)),counts);return torch.log1p(result)/self.scale

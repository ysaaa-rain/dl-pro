"""Off-diagonal structural terms using train-fitted distance expectations."""
import numpy as np
import torch
from torch.nn import functional as F


def masked_ssim_loss(a,b,mask):
    weights=mask.to(a.dtype)[None,None];den=F.avg_pool2d(weights,7,stride=1,padding=3).clamp_min(1e-6)
    def average(x):return F.avg_pool2d(x*weights,7,stride=1,padding=3)/den
    ma,mb=average(a),average(b);va=(average(a*a)-ma*ma).clamp_min(0);vb=(average(b*b)-mb*mb).clamp_min(0);cov=average(a*b)-ma*mb
    # Fixed log-OE dynamic range4; no test-derived normalization.
    c1=.04**2;c2=.12**2;ssim=((2*ma*mb+c1)*(2*cov+c2))/((ma*ma+mb*mb+c1)*(va+vb+c2))
    return (1-ssim[:,:,mask].clamp(-1,1)).mean()


def structure_terms(predicted_counts,target_counts,expectations):
    terms={}
    for bp,profile in expectations.items():
        factor=int(bp)//10;n=predicted_counts.shape[-1]//factor
        a=F.avg_pool2d(predicted_counts,factor);b=F.avg_pool2d(target_counts,factor)
        distance=abs(torch.arange(n,device=a.device)[:,None]-torch.arange(n,device=a.device)[None]);expect=torch.as_tensor(profile,device=a.device,dtype=a.dtype)[distance][None,None].clamp_min(1e-5)
        a=torch.log1p(a/expect);b=torch.log1p(b/expect);mask=distance*int(bp)>=200
        terms[int(bp)]=(F.mse_loss(a[:,:,mask],b[:,:,mask]),masked_ssim_loss(a,b,mask))
    return terms


def fit_expectations(raw,indices):
    output={}
    for bp in [50,100,160]:
        factor=bp//10;n=raw.shape[-1]//factor;total=np.zeros(n,dtype=np.float64);den=np.zeros(n,dtype=np.float64);distance=abs(np.arange(n)[:,None]-np.arange(n)[None,:]).ravel()
        for idx in indices:
            a=raw[idx,:n*factor,:n*factor].astype('float32').reshape(n,factor,n,factor).mean((1,3));total+=np.bincount(distance,weights=a.ravel(),minlength=n);den+=np.bincount(distance,minlength=n)
        output[bp]=np.maximum(total/den,1e-5).astype('float32')
    return output

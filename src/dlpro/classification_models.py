"""Class-imbalance experiments using two physical scales and central context."""
import torch
from torch import nn
from torch.nn import functional as F


class ScaleAwareClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.branches=nn.ModuleList([nn.Sequential(nn.Conv2d(1,16,5,2,2),nn.ReLU(),nn.Conv2d(16,24,3,2,1),nn.ReLU(),nn.Conv2d(24,32,3,2,1),nn.ReLU()) for _ in range(2)])
        self.head=nn.Sequential(nn.Linear(2*(32*16+64),64),nn.ReLU(),nn.Dropout(.25),nn.Linear(64,3))
    def forward(self,x):
        features=[]
        for channel,branch in enumerate(self.branches):
            z=branch(x[:,channel:channel+1]);h,w=z.shape[-2:]
            features.append(torch.cat([F.adaptive_avg_pool2d(z,(4,4)).flatten(1),z.mean((-2,-1)),z[:,:,h//4:3*h//4,w//4:3*w//4].mean((-2,-1))],1))
        return self.head(torch.cat(features,1))


def reverse_genome(x,reverse):
    """Reverse both genome axes together; never break the contact diagonal."""
    return torch.where(reverse[:,None,None,None],torch.flip(x,(-2,-1)),x)


class FineScaleClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.branches=nn.ModuleList([nn.Sequential(nn.Conv2d(1,16,5,2,2),nn.ReLU(),nn.Conv2d(16,24,3,2,1),nn.ReLU(),nn.Conv2d(24,32,3,2,1),nn.ReLU()) for _ in range(3)])
        self.embedding=nn.Sequential(nn.Linear(3*(32*16+64),96),nn.ReLU(),nn.Dropout(.2))
        self.head=nn.Linear(96,3)
    def encode(self,x):
        features=[]
        for channel,branch in enumerate(self.branches):
            z=branch(x[:,channel:channel+1]);h,w=z.shape[-2:];mean=z.mean((-2,-1));center=z[:,:,h//4:3*h//4,w//4:3*w//4].mean((-2,-1))
            features.append(torch.cat([F.adaptive_avg_pool2d(z,(4,4)).flatten(1),mean,center-mean],1))
        return self.embedding(torch.cat(features,1))
    def forward(self,x):return self.head(self.encode(x))


def repeat_js_divergence(logits):
    """First half and second half contain matched replicate views of each locus."""
    if len(logits)%2:raise ValueError('Paired view batch must have even size')
    logp=F.log_softmax(logits,1);p=logp.exp();n=len(p)//2;mixture=(p[:n]+p[n:])/2;logm=mixture.clamp_min(1e-8).log()
    return .5*((p[:n]*(logp[:n]-logm)).sum(1)+(p[n:]*(logp[n:]-logm)).sum(1)).mean()


class SpanAwareClassifier(FineScaleClassifier):
    def __init__(self):
        super().__init__();self.span_head=nn.Linear(96,1)
        nn.init.zeros_(self.span_head.weight);nn.init.zeros_(self.span_head.bias)
    def forward(self,x):
        features=self.encode(x);return self.head(features),self.span_head(features)[:,0]

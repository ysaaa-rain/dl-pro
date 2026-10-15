"""Cross-repeat representation learning without annotation targets."""
import torch
from torch.nn import functional as F


def paired_info_nce(first,second,temperature=.2):
    if first.shape!=second.shape or first.ndim!=2 or len(first)<2:raise ValueError('Need matched NxD views with at least two locations')
    if temperature<=0:raise ValueError('Temperature must be positive')
    logits=F.normalize(first,dim=1)@F.normalize(second,dim=1).T/temperature
    target=torch.arange(len(first),device=first.device)
    return .5*(F.cross_entropy(logits,target)+F.cross_entropy(logits.T,target))

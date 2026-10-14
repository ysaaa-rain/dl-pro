import numpy as np
import torch
from torch.nn import functional as F
from dlpro.refinement import CountConsistentSR,project_counts,morphology_features


def test_projection_conserves_observed_counts_and_gradients():
    torch.manual_seed(4)
    low=torch.rand(2,1,12,12);low=(low+low.transpose(-1,-2))/2;low[:,:,2,2]=0
    prior=torch.rand(2,1,48,48,requires_grad=True)
    high=project_counts(prior,low)
    assert torch.allclose(F.avg_pool2d(high,4),low,atol=2e-7)
    assert torch.allclose(high,high.transpose(-1,-2),atol=2e-7)
    high.square().mean().backward();assert torch.isfinite(prior.grad).all()
    model=CountConsistentSR(2.5);prediction=model(torch.log1p(low)/2.5)
    assert torch.allclose(F.avg_pool2d(torch.expm1(prediction*2.5),4),low,atol=2e-7)


def test_morphology_amplitude_invariance_and_coherent_structure():
    raw=np.ones((2,2,64,64),dtype='float32');raw[1,:,24:40,24:40]=12
    expected=np.ones((2,64),dtype='float32')
    f=morphology_features(raw,expected)
    assert np.allclose(f,morphology_features(raw*10,expected),atol=2e-6)
    assert f[1].max()>f[0].max()+.1
    zero=morphology_features(np.zeros_like(raw),expected)
    assert np.isfinite(zero).all() and np.all(zero==0)

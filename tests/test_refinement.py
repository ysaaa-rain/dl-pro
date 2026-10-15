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


def test_support_mask_removes_partial_row_artifact():
    raw=np.ones((1,2,64,64));raw[:,:,32,:]*=.01;raw[:,:,:,32]*=.01
    rows=np.ones((1,2,64),bool);rows[:,:,32]=False;expected=np.ones((2,64))
    unmasked=morphology_features(raw,expected)
    masked=morphology_features(raw,expected,rows)
    assert unmasked.max()>1e-4 and masked.max()<1e-10


def test_native_support_weights_the_short_final_bin():
    from dlpro.coverage_support import coarse_support
    from dlpro.experiment_core import LENGTH
    coverage=np.ones((LENGTH+9)//10,dtype='uint8');coverage[-1]=0
    fractions=coarse_support(coverage,[LENGTH-25])
    assert fractions.shape==(1,2,128)
    for channel in range(2):
        assert np.isclose(fractions[0,channel].min(),50/52)
        assert np.sum(fractions[0,channel]<1)==1


def test_context_sr_conserves_measured_counts_and_gradients():
    from dlpro.refinement import ContextCountSR
    import torch
    from torch.nn import functional as F
    torch.manual_seed(9)
    low=torch.rand(1,1,8,8);low=(low+low.transpose(-1,-2))/2;low[:,:,0,0]=0
    model=ContextCountSR(2.5);prediction=model(torch.log1p(low)/2.5)
    decoded=torch.expm1(prediction*2.5)
    assert torch.allclose(F.avg_pool2d(decoded,4),low,atol=1e-6)
    assert torch.allclose(decoded,decoded.transpose(-1,-2),atol=1e-6)
    prediction.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())

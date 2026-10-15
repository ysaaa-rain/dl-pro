import numpy as np
import pytest
import torch
from dlpro.structure_losses import structure_terms,masked_ssim_loss
from dlpro.self_supervision import paired_info_nce


def test_masked_structure_losses_ignore_forbidden_diagonal_and_backpropagate():
    torch.manual_seed(2);a=torch.rand(2,1,20,20,requires_grad=True);b=a.detach().clone();distance=abs(torch.arange(20)[:,None]-torch.arange(20)[None]);mask=distance>=3
    b[:,:,~mask]=500
    loss=masked_ssim_loss(a,b,mask)
    assert abs(float(loss.detach()))<1e-5
    (loss+a.square().mean()).backward();assert torch.isfinite(a.grad).all()
    counts=torch.rand(1,1,128,128,requires_grad=True);expect={bp:np.ones(128//(bp//10),dtype='float32') for bp in [50,100,160]}
    terms=structure_terms(counts,counts.detach()+.1,expect);total=sum(mse+ssim for mse,ssim in terms.values());assert torch.isfinite(total);total.backward();assert torch.isfinite(counts.grad).all()


def test_cross_repeat_infonce_correspondence_and_symmetry():
    identity=torch.eye(8,requires_grad=True);loss=paired_info_nce(identity,identity)
    assert loss<paired_info_nce(identity,identity.flip(0))
    assert torch.allclose(loss,paired_info_nce(identity,identity))
    loss.backward();assert torch.isfinite(identity.grad).all()
    with pytest.raises(ValueError):paired_info_nce(torch.ones(1,4),torch.ones(1,4))

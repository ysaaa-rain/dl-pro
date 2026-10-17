import numpy as np
import torch
from dlpro.targeted_review import shape_input, masked_errors, supervision_repeats, same_objective
from dlpro.allocation_model import ConditionalDistributionSR
from dlpro.ridge_features import directional_features
from dlpro.refinement import morphology_features


def test_supported_shape_amplitude_and_missing_pixels():
    rng=np.random.default_rng(4)
    raw=rng.random((3,16,16));raw=(raw+raw.transpose(0,2,1))/2
    rows=np.ones((3,16),bool);rows[:,3]=False
    expected=np.ones(16)
    a,mask,scale=shape_input(raw,rows,expected)
    b,_,_=shape_input(raw*10,rows,expected,scale)
    np.testing.assert_allclose(a,b,atol=1e-6)
    corrupt=raw.copy();corrupt[:,3,:]=1e8;corrupt[:,:,3]=1e8
    c,_,_=shape_input(corrupt,rows,expected,scale)
    np.testing.assert_allclose(a,c)
    assert not mask[:,:,3].any()


def test_masked_unique_loss_ignores_unsupported_pixels():
    target=torch.zeros(1,1,8,8)
    mask=torch.ones_like(target,dtype=torch.bool);mask[:,:,3]=False;mask[:,:,:,3]=False
    pred=torch.ones_like(target,requires_grad=True)
    loss=masked_errors(pred,target,mask).mean();loss.backward()
    assert loss == 1 and pred.grad[:,:,3].abs().sum() == 0
    assert torch.isfinite(pred.grad).all()


def test_matched_repeat_sampling_and_same_loss():
    assert (supervision_repeats('allocation_single_same',np.random.default_rng(4),100)==0).all()
    assert set(supervision_repeats('allocation_shared_same',np.random.default_rng(4),100))=={0,1}
    cfg=dict(factor=4,channels=2,dilations=[1],native_bp=10,distance_bands_bp=[0,200,1000])
    model=ConditionalDistributionSR(2.,np.ones(16),cfg)
    high=torch.zeros(1,1,16,16);high[:,:,0,1]=high[:,:,1,0]=3
    loss=same_objective(model,high,cfg);loss.backward()
    assert torch.isfinite(loss) and model.output[0].weight.grad.abs().sum()>0


def test_direction_features_subset_does_not_change_values():
    rng=np.random.default_rng(17);raw=rng.random((4,2,32,32)).astype('float32')
    raw=(raw+raw.transpose(0,1,3,2))/2
    rows=np.ones((4,2,32),bool);rows[:,:,5]=False
    expected=np.ones((2,32));ids=np.array([0,3])
    np.testing.assert_array_equal(directional_features(raw,expected,rows)[ids],directional_features(raw[ids],expected,rows[ids]))
    np.testing.assert_array_equal(morphology_features(raw,expected,rows)[ids],morphology_features(raw[ids],expected,rows[ids]))

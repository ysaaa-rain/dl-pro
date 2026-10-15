import numpy as np
import torch
from dlpro.classification_models import reverse_genome
from dlpro.shape_features import forest_probabilities


def test_genome_reversal_preserves_symmetry_and_diagonal_gradients():
    x=torch.arange(2*2*8*8,dtype=torch.float32).reshape(2,2,8,8);x=((x+x.transpose(-1,-2))/2).requires_grad_()
    result=reverse_genome(x,torch.tensor([True,False]))
    assert torch.equal(result[1],x[1])
    assert torch.equal(result[0].diagonal(dim1=-2,dim2=-1),x[0].diagonal(dim1=-2,dim2=-1).flip(-1))
    assert torch.equal(result,result.transpose(-1,-2))
    result.sum().backward();assert torch.equal(x.grad,torch.ones_like(x))


def test_portable_forest_uses_weighted_leaf_probabilities():
    arrays={'0_left':np.array([1,-1,-1]),'0_right':np.array([2,-1,-1]),'0_feature':np.array([0,-2,-2]),'0_threshold':np.array([.5,-2,-2]),'0_value':np.array([[[2,2,4]],[[3,1,0]],[[0,0,4]]])}
    result=forest_probabilities(arrays,np.array([[.2],[.8],[.5]]))
    assert np.allclose(result,[[.75,.25,0],[0,0,1],[.75,.25,0]])


def test_repeat_js_is_symmetric_shift_invariant_and_differentiable():
    from dlpro.classification_models import repeat_js_divergence
    torch.manual_seed(7)
    a=torch.randn(4,3);b=torch.randn(4,3);logits=torch.cat([a,b]).requires_grad_()
    score=repeat_js_divergence(logits)
    assert 0<=score.item()<=np.log(2)
    assert torch.allclose(score,repeat_js_divergence(torch.cat([b,a])),atol=1e-7)
    assert torch.allclose(score,repeat_js_divergence(logits+100),atol=2e-6)
    assert abs(repeat_js_divergence(torch.cat([a,a])).item())<1e-7
    score.backward();assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum()>0


def test_span_head_preserves_loaded_classification_backbone():
    from dlpro.classification_models import FineScaleClassifier,SpanAwareClassifier
    import torch
    torch.manual_seed(3)
    original=FineScaleClassifier().eval();auxiliary=SpanAwareClassifier().eval()
    missing=auxiliary.load_state_dict(original.state_dict(),strict=False)
    assert set(missing.missing_keys)=={'span_head.weight','span_head.bias'}
    x=torch.randn(2,3,32,32)
    with torch.no_grad():before=original(x);after,span=auxiliary(x)
    assert torch.equal(before,after)
    assert torch.equal(span,torch.zeros_like(span))

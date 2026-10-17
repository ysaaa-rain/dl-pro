import numpy as np
import torch

from dlpro.allocation_model import allocation_probabilities, ConditionalDistributionSR, unique_allocation_nll
from dlpro.method_review import interior_mask
from dlpro.sr_review import exact_count_log_quantile
from dlpro.refinement import project_counts


def test_probabilities_and_expectation_mass_symmetry_zero_and_context():
    cfg = dict(factor=4, channels=4, dilations=[1, 2])
    model = ConditionalDistributionSR(2, np.ones(32), cfg)
    low = torch.rand(1, 1, 8, 8)
    low = (low + low.transpose(-1, -2)) / 2
    low[..., 0, :] = 0
    low[..., :, 0] = 0
    pred, probability = model(low)
    assert torch.allclose(torch.nn.functional.avg_pool2d(pred, 4), low, atol=1e-6)
    assert torch.allclose(pred, pred.transpose(-1, -2), atol=1e-6)
    assert (pred >= 0).all() and pred[..., :4, :].sum() == 0
    assert torch.allclose(torch.nn.functional.avg_pool2d(probability, 4), torch.full_like(low, 1 / 16))
    torch.nn.init.normal_(model.output[0].weight, std=.1)
    modified = low.clone()
    modified[..., 2, 3] += 1
    modified[..., 3, 2] += 1
    assert not torch.allclose(model(low)[1], model(modified)[1])


def test_unique_diagonal_observation_and_zero_source_probability_gradient():
    logits = torch.zeros(1, 1, 8, 8, dtype=torch.double, requires_grad=True)
    target = torch.zeros_like(logits)
    target[..., 0, 1] = target[..., 1, 0] = 3
    probability = allocation_probabilities(logits, 4)
    loss = unique_allocation_nll(probability, target, 4, 10, [0, 10000])
    assert torch.allclose(loss, torch.tensor(np.log(10), dtype=torch.double))
    gradient = torch.autograd.grad(loss, logits)[0]
    assert gradient.abs().sum() > 0 and torch.isfinite(gradient).all()
    # Changing the mirrored lower target cannot change the unique-observation loss.
    changed = target.clone()
    changed[..., 1, 0] = 300
    assert torch.allclose(loss, unique_allocation_nll(probability, changed, 4, 10, [0, 10000]))
    assert unique_allocation_nll(probability, target * 0, 4, 10, [0, 10000]) == 0


def test_unique_probability_gradient_matches_numerical_derivative():
    logits = torch.randn(1, 1, 8, 8, dtype=torch.double, requires_grad=True)
    target = torch.arange(64, dtype=torch.double).reshape(1, 1, 8, 8) % 3
    function = lambda z: unique_allocation_nll(allocation_probabilities(z, 4), target, 4, 10, [0, 10000])
    assert torch.autograd.gradcheck(function, (logits,), eps=1e-6, atol=1e-4)


def test_spatial_guard_rejects_shared_boundary_and_wraps():
    assert interior_mask([0, 95000, -100, 100000], [10000, 105000, 10000, 110000], 100000, 4641652).tolist() == [True, False, False, True]


def test_streaming_quantile_matches_every_training_pixel_and_excludes_holdout():
    raw = [np.arange(48, dtype='uint32').reshape(3, 4, 4),
           np.arange(48, dtype='uint32').reshape(3, 4, 4) ** 2]
    factors, ids = [1., .7], [0, 1]
    expected = np.concatenate([np.log1p(a[ids].astype(float) * m).ravel() for a, m in zip(raw, factors)])
    for q in (0, .25, .995, 1):
        assert np.isclose(exact_count_log_quantile(raw, ids, factors, q), np.quantile(expected, q))
    raw[0][2] = 1000000
    assert np.isclose(exact_count_log_quantile(raw, ids, factors, .995), np.quantile(expected, .995))


def test_160bp_aligned_shape_is_conservation_control_not_learned_information():
    high = torch.rand(1, 1, 32, 32)
    high = (high + high.transpose(-1, -2)) / 2
    low = torch.nn.functional.avg_pool2d(high, 4)
    prediction = project_counts(torch.rand_like(high), low)
    assert torch.allclose(torch.nn.functional.avg_pool2d(prediction, 16),
                          torch.nn.functional.avg_pool2d(high, 16), atol=1e-6)

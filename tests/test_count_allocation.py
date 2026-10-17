import torch
from dlpro.count_allocation import conditional_allocation_nll


def test_count_likelihood_prefers_observed_allocation_ignores_empty_targets_and_has_gradients():
    target = torch.zeros(1, 1, 32, 32)
    target[0, 0, 0, 24] = 7
    target[0, 0, 24, 0] = 7
    low = torch.ones(1, 1, 8, 8)
    uniform = torch.ones_like(target, requires_grad=True)
    matched = torch.ones_like(target)
    matched[0, 0, 0, 24] = 100
    matched[0, 0, 24, 0] = 100
    assert conditional_allocation_nll(matched, target, low) < conditional_allocation_nll(uniform, target, low)
    loss = conditional_allocation_nll(uniform, target, low)
    loss.backward()
    assert torch.isfinite(uniform.grad).all()
    assert uniform.grad[0, 0, 0, 24] < 0
    assert uniform.grad[0, 0, 12:16, 12:16].abs().sum() == 0
    assert conditional_allocation_nll(uniform.detach(), torch.zeros_like(target), low) == 0

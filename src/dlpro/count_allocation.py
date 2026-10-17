"""Composite count likelihood for observable within-40bp-block allocation."""

import torch


def conditional_allocation_nll(prediction, target, source_low):
    """Equal near/far count-normalized NLL, skipping unobserved target blocks.

    Symmetric contact pixels are dependent. This is a composite objective,
    not an exact independent-read likelihood or a significance calculation.
    """
    batch, channels, height, width = prediction.shape
    if channels != 1 or height != width or height % 4:
        raise ValueError("Expected Bx1x(4n)x(4n) symmetric contact images")
    n = height // 4
    def blocks(image):
        return image.reshape(batch, 1, n, 4, n, 4).permute(0, 1, 2, 4, 3, 5)
    values = blocks(prediction).clamp_min(1e-12)
    probability = values / values.sum((-2, -1), keepdim=True)
    counts = blocks(target)
    total = counts.sum((-2, -1))
    nll = -(counts * probability.log()).sum((-2, -1))
    distance = torch.arange(n, device=prediction.device)[None, :] - torch.arange(n, device=prediction.device)[:, None]
    eligible = (source_low > 0) & (total > 0) & (distance[None, None] >= 0)
    sections = []
    for mask in (distance < 5, distance >= 5):
        valid = eligible & mask[None, None]
        sections.append((nll * valid).sum() / (total * valid).sum().clamp_min(1))
    return torch.stack(sections).mean()

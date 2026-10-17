"""Predict within-block probabilities independently of measured source mass."""

import torch
from torch import nn
from torch.nn import functional as F

from .refinement import DilatedResidual


def block_view(image, factor):
    b, c, h, w = image.shape
    if h != w or h % factor:
        raise ValueError("Square image dimensions must be divisible by factor")
    n = h // factor
    return image.reshape(b, c, n, factor, n, factor).permute(0, 1, 2, 4, 3, 5)


def image_view(blocks):
    b, c, n, _, f, _ = blocks.shape
    return blocks.permute(0, 1, 2, 4, 3, 5).reshape(b, c, n * f, n * f)


def allocation_probabilities(logits, factor):
    logits = (logits + logits.transpose(-1, -2)) / 2
    blocks = block_view(logits, factor)
    shape = blocks.shape
    return image_view(blocks.flatten(-2).softmax(-1).reshape(shape))


def unique_allocation_nll(probability, target, factor, native_bp, distance_bands_bp):
    """Target-total conditional cross entropy on unique upper-triangle contacts.

    A coarse diagonal block contains only its native upper-triangle observations;
    their probabilities are renormalized there. Zero source blocks are included:
    probability is latent and does not depend on multiplication by source mass.
    Correlated biological blocks make this a composite training objective, not an
    independent-read significance model. Empty target blocks carry no allocation
    observation; their raw reconstruction error remains in evaluation.
    """
    p = block_view(probability, factor)
    y = block_view(target, factor)
    n = p.shape[2]
    ix = torch.arange(n, device=p.device)
    delta = ix[None, :] - ix[:, None]
    sub = torch.arange(factor, device=p.device)
    unique = ((delta[:, :, None, None] > 0)
              | ((delta[:, :, None, None] == 0) & (sub[None, :] >= sub[:, None])))
    unique = unique[None, None]
    normalized = p / (p * unique).sum((-2, -1), keepdim=True).clamp_min(1e-12)
    count = (y * unique).sum((-2, -1))
    loss = -(y * unique * normalized.clamp_min(1e-12).log()).sum((-2, -1))
    sections = []
    bp = delta * factor * native_bp
    for lower, upper in zip(distance_bands_bp[:-1], distance_bands_bp[1:]):
        mask = ((bp >= lower) & (bp < upper))[None, None]
        sections.append((loss * mask).sum() / (count * mask).sum().clamp_min(1))
    return torch.stack(sections).mean()


class ConditionalDistributionSR(nn.Module):
    """Source-context distribution + train-only distance prior, then mass allocation."""

    def __init__(self, scale, profile, config):
        super().__init__()
        self.scale = float(scale)
        self.factor = int(config["factor"])
        channels = int(config["channels"])
        self.encoder = nn.Sequential(nn.Conv2d(1, channels, 3, padding=1), nn.ReLU(),
                                     *[DilatedResidual(channels, d) for d in config["dilations"]])
        self.output = nn.Sequential(nn.Conv2d(channels, self.factor ** 2, 3, padding=1),
                                    nn.PixelShuffle(self.factor))
        nn.init.zeros_(self.output[0].weight)
        nn.init.zeros_(self.output[0].bias)
        self.register_buffer("profile", torch.as_tensor(profile, dtype=torch.float32))

    def forward(self, low_counts):
        correction = self.output(self.encoder(torch.log1p(low_counts) / self.scale))
        n = correction.shape[-1]
        ix = torch.arange(n, device=low_counts.device)
        prior = self.profile[(ix[:, None] - ix[None]).abs()].clamp_min(1e-8).log()[None, None]
        probability = allocation_probabilities(prior + correction, self.factor)
        mass = F.interpolate(low_counts, scale_factor=self.factor, mode="nearest") * self.factor ** 2
        return probability * mass, probability

"""Focused background reconstruction and matched same-repeat supervision."""

import numpy as np
import torch
from torch.nn import functional as F
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .allocation_model import unique_allocation_nll


def shape_input(raw, rows, expected, scale=None, quantile=.995):
    """Supported, distance-normalized, amplitude-invariant single-channel image."""
    n = raw.shape[-1]
    d = abs(np.arange(n)[:, None] - np.arange(n)[None])
    mask = (d >= 2)[None] & rows[:, :, None] & rows[:, None, :]
    oe = raw / expected[d][None]
    amplitude = (oe * mask).sum((1, 2)) / mask.sum((1, 2)).clip(1)
    image = np.where(mask, np.log1p(oe / amplitude[:, None, None].clip(1e-12)), 0)
    if scale is None:
        scale = max(float(np.quantile(image[mask], quantile)), 1e-6)
    return (image / scale).clip(0, 1).astype('float32')[:, None], mask[:, None], scale


def masked_errors(prediction, target, mask):
    """One loss per window; upper triangle counts each supported pair once."""
    n = target.shape[-1]
    triangle = torch.triu(torch.ones((n, n), dtype=torch.bool, device=target.device), 2)
    keep = mask & triangle
    symmetric = (prediction + prediction.transpose(-1, -2)) / 2
    return ((symmetric-target).square()*keep).sum((1, 2, 3)) / keep.sum((1, 2, 3)).clamp_min(1)


class NuisanceResidual:
    """Fit nuisance regression on training backgrounds only; no held-out fitting."""
    def __init__(self, alpha):
        self.scaler = StandardScaler()
        self.model = Ridge(alpha=alpha)

    def fit(self, nuisance, response):
        x = self.scaler.fit_transform(nuisance)
        self.model.fit(x, response)
        residual = response - self.model.predict(x)
        self.location = np.median(residual, axis=0)
        self.spread = np.maximum(1.4826*np.median(abs(residual-self.location), axis=0), .05)
        return self

    def predict(self, nuisance, response):
        return (response-self.model.predict(self.scaler.transform(nuisance))-self.location)/self.spread


def same_objective(model, high, sr):
    low = F.avg_pool2d(high, sr['factor'])
    _, probability = model(low)
    return unique_allocation_nll(probability, high, sr['factor'], sr['native_bp'], sr['distance_bands_bp'])


def supervision_repeats(arm, rng, size):
    if arm == 'allocation_single_same':
        return np.zeros(size, dtype=int)
    if arm == 'allocation_shared_same':
        return rng.integers(0, 2, size)
    raise ValueError(f'Unknown matched same-repeat arm: {arm}')


def pearson_all(a, b):
    """Undefined correlation is explicit, never replaced with perfect agreement."""
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])

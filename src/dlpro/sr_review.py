"""One data and objective definition shared by review training and evaluation."""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from .allocation_model import ConditionalDistributionSR, unique_allocation_nll
from .experiment_core import SubpixelSR
from .refinement import ContextCountSR, project_counts
from .structure_losses import structure_terms


def exact_count_log_quantile(raw, indices, multipliers, quantile):
    """Exact pooled pixel quantile from streaming integer-count histograms."""
    values, weights = [], []
    for source, multiplier in zip(raw, multipliers):
        histogram = np.zeros(1, dtype="int64")
        for index in indices:
            image = np.asarray(source[index], dtype="int64")
            counts = np.bincount(image.ravel())
            if len(counts) > len(histogram):
                histogram = np.pad(histogram, (0, len(counts) - len(histogram)))
            histogram[:len(counts)] += counts
        observed = np.flatnonzero(histogram)
        values.extend(np.log1p(observed * multiplier))
        weights.extend(histogram[observed])
    values, weights = np.asarray(values), np.asarray(weights, dtype="int64")
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cumulative = weights.cumsum()
    position = (cumulative[-1] - 1) * quantile
    lower, upper = int(np.floor(position)), int(np.ceil(position))
    a, b = values[np.searchsorted(cumulative, [lower, upper], side="right")]
    return float(a + (b - a) * (position - lower))


def load_data(cfg):
    root = Path(cfg["output"]) / "data"
    plan = pd.read_csv(root / "sr_windows.csv")
    manifest = json.loads((root / "preparation.json").read_text(encoding="utf-8"))
    raw = [np.load(Path(cfg["cache"]) / f"raw_rep{r}.npy", mmap_mode="r") for r in (1, 2)]
    libraries = manifest["library_sizes"]
    depth = [1., libraries["1"] / libraries["2"]]
    return plan, raw, depth


def make_batch(raw, depth, indices, repeats, reverse=None):
    source = np.stack([raw[int(r)][int(i)].astype("float32") * depth[int(r)] for i, r in zip(indices, repeats)])
    other = np.stack([raw[1-int(r)][int(i)].astype("float32") for i, r in zip(indices, repeats)])
    high = torch.from_numpy(source[:, None])
    target = torch.from_numpy(other[:, None])
    if reverse is not None:
        high[reverse] = high[reverse].flip((-2, -1))
        target[reverse] = target[reverse].flip((-2, -1))
    return high, target


def make_model(arm, scale, profile, sr):
    if arm.startswith("legacy_"):
        return SubpixelSR()
    if arm.startswith("context_"):
        return ContextCountSR(scale, channels=sr["channels"], dilations=sr["dilations"])
    return ConditionalDistributionSR(scale, profile, sr)


def infer_model(model, arm, low, scale, sr):
    probability = unconstrained = None
    if arm.startswith("allocation_"):
        counts, probability = model(low)
    elif arm.startswith("legacy_"):
        unconstrained = model((torch.log1p(low) / scale).clamp(0, 1))
        counts = project_counts(torch.expm1(unconstrained.clamp(0, 1) * scale), low)
    else:
        counts = torch.expm1(model(torch.log1p(low) / scale) * scale)
    return counts, probability, unconstrained


def objective(model, arm, high, cross_raw, scale, expectations, sr):
    low = F.avg_pool2d(high, sr["factor"])
    counts, probability, unconstrained = infer_model(model, arm, low, scale, sr)
    target_log = (torch.log1p(high) / scale).clamp(0, 1)
    if arm.startswith("legacy_"):
        return F.mse_loss(unconstrained, target_log)
    if arm == "allocation_expanded_cross":
        return unique_allocation_nll(probability, cross_raw, sr["factor"], sr["native_bp"], sr["distance_bands_bp"])
    native = F.mse_loss((torch.log1p(counts) / scale).clamp(0, 1), target_log)
    terms = structure_terms(counts, high, expectations)
    shape_mse = torch.stack([v[0] for v in terms.values()]).mean()
    shape_ssim = torch.stack([v[1] for v in terms.values()]).mean()
    value = native + sr["morphology_mse_weight"] * shape_mse + sr["morphology_ssim_weight"] * shape_ssim
    if arm == "allocation_expanded_mixed":
        value = value + sr["mixed_nll_weight"] * unique_allocation_nll(
            probability, cross_raw, sr["factor"], sr["native_bp"], sr["distance_bands_bp"])
    return value

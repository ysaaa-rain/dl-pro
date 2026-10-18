"""Recompute the selected statistical model and metric backgrounds from train only."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
import torch

from dlpro.method_review import load_config, sha256, write_json
from dlpro.sr_review import exact_count_log_quantile, load_data
from dlpro.structure_losses import fit_expectations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/method_review_final.json")
    args = parser.parse_args()
    cfg = load_config(args.config)
    sr = cfg["sr"]
    root = Path(cfg["output"]) / sr.get("result_subdir", "sr")
    plan, raw, depth = load_data(cfg)
    train = np.flatnonzero(plan.split == "train")
    saved = np.load(root / "normalization.npz")
    assert np.array_equal(saved["training_indices"], train)
    quantile = exact_count_log_quantile(raw, train, depth, sr["normalization_quantile"])
    assert quantile == float(saved["scale"])
    n = sr["patch_bins"]
    distance = abs(np.arange(n)[:, None] - np.arange(n)[None]).ravel()
    denominator = np.bincount(distance, minlength=n) * len(train) * len(raw)
    totals = np.zeros(n)
    for a, multiplier in zip(raw, depth):
        for index in train:
            totals += np.bincount(distance, weights=a[index].astype(float).ravel() * multiplier, minlength=n)
    direct = totals / denominator
    direct = np.exp(gaussian_filter1d(np.log(direct.clip(1e-8)), sr["prior_smoothing_sigma"]))
    assert np.allclose(direct, saved["profile"], rtol=1e-6, atol=1e-8)
    recalculated = [fit_expectations(a, train) for a in raw]
    errors = {}
    for bp in sr["morphology_scales_bp"]:
        expected = sum(item[bp] * multiplier for item, multiplier in zip(recalculated, depth)) / len(raw)
        errors[str(bp)] = float(abs(expected - saved[f"expected_{bp}"]).max())
        assert np.allclose(expected, saved[f"expected_{bp}"], rtol=1e-6, atol=1e-8)
    lock = json.loads((root / "evaluation/test_lock.json").read_text(encoding="utf-8"))
    for arm in sr["arms"]:
        for seed in sr["train_seeds"]:
            path = root / f"{arm}_seed{seed}.pt"
            assert sha256(path) == lock["checkpoint_sha256"][path.stem]
            checkpoint = torch.load(path, weights_only=True)
            assert np.array_equal(checkpoint["profile"].numpy(), saved["profile"].astype('float32'))
            assert checkpoint["scale"] == quantile
    write_json(Path(cfg["output"]) / "train_prior_audit.json", dict(
        training_indices=train.tolist(), training_groups=sorted(plan.iloc[train].group.unique().astype(int).tolist()),
        exact_train_log_quantile=quantile, normalization_sha256=sha256(root / "normalization.npz"),
        independently_accumulated_profile_max_absolute_error=float(abs(direct-saved["profile"]).max()),
        oe_expectation_max_absolute_errors=errors,
        profile_matches_every_pretest_locked_checkpoint=True,
        interpretation="Recomputed train-only parameters; not a new test performance run"))
    print("train-only quantile, independently accumulated distance prior and all OE backgrounds verified", flush=True)


if __name__ == "__main__":
    main()

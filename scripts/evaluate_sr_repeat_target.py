"""Evaluate the frozen cross-repeat-target model on the original spatial test blocks."""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from dlpro.experiment_core import save_json
from dlpro.count_allocation import conditional_allocation_nll
from dlpro.refinement import ContextCountSR, project_counts
from dlpro.structure_losses import structure_terms
from dlpro.superresolution import scores


OUT = Path("results/sr_repeat_target")
METHODS = ("projected_bicubic", "sparse_matched", "dense_multiscale", "cross_repeat_target")


def load_model(path):
    checkpoint = torch.load(path, weights_only=True, map_location="cpu")
    model = ContextCountSR(float(checkpoint["scale"])).eval()
    model.load_state_dict(checkpoint["state_dict"])
    return model, float(checkpoint["scale"])


def infer(high, models, scale):
    high = torch.from_numpy(high[None, None].astype("float32"))
    low = F.avg_pool2d(high, 4)
    result = {"projected_bicubic": project_counts(
        F.interpolate(low, scale_factor=4, mode="bicubic", align_corners=False), low
    )}
    with torch.no_grad():
        for name, model in models.items():
            result[name] = torch.expm1(model(torch.log1p(low) / scale) * scale)
    for name, pred in result.items():
        assert torch.isfinite(pred).all() and pred.min() >= 0, name
        assert (F.avg_pool2d(pred, 4) - low).abs().max() < 1e-4, name
    return high, low, result


def metrics(pred, target, scale, expectation):
    a = (torch.log1p(pred) / scale).clamp(0, 1).numpy()[0, 0]
    b = (torch.log1p(target) / scale).clamp(0, 1).numpy()[0, 0]
    image = scores(b, a)
    shape = structure_terms(pred, target, {100: expectation})[100][1]
    return dict(**image, oe_100bp_masked_ssim=float(1 - shape))


def common_objective(pred, target, scale, expectations):
    native = F.mse_loss((torch.log1p(pred) / scale).clamp(0, 1),
                        (torch.log1p(target) / scale).clamp(0, 1))
    terms = structure_terms(pred, target, expectations)
    mse = torch.stack([entry[0] for entry in terms.values()]).mean()
    ssim = torch.stack([entry[1] for entry in terms.values()]).mean()
    return float(native + .1 * mse + .05 * ssim)


def group_interval(table, metric, left, right):
    panel = table.pivot(index=["start", "group"], columns="method", values=metric)
    delta = (panel[left] - panel[right]).to_numpy()
    groups = panel.index.get_level_values("group").to_numpy()
    distinct = np.unique(groups)
    rng = np.random.default_rng(2031)
    estimates = []
    for _ in range(2000):
        draw = rng.choice(distinct, len(distinct), replace=True)
        indices = np.concatenate([np.flatnonzero(groups == item) for item in draw])
        estimates.append(float(delta[indices].mean()))
    return dict(delta=float(delta.mean()), interval95=np.quantile(estimates, [0.025, 0.975]).tolist(),
                groups=len(distinct))


def main():
    global OUT, METHODS
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditional", action="store_true")
    args = parser.parse_args()
    if args.conditional:
        OUT = Path("results/sr_conditional")
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    models = {}
    scale = None
    locations = [
        ("sparse_matched", "results/sr_data_controls/sparse_matched_seed42.pt"),
        ("dense_multiscale", "results/sr_data_controls/dense_multiscale_seed42.pt"),
        ("cross_repeat_target", "results/sr_repeat_target/model_seed42.pt"),
    ]
    if args.conditional:
        locations.extend((name, OUT / f"{name}_seed42.pt")
                         for name in ("conditional_rep1", "conditional_rep2"))
    METHODS = ("projected_bicubic", *[name for name, _ in locations])
    for name, path in locations:
        models[name], current = load_model(path)
        if scale is not None:
            assert current == scale
        scale = current
    with np.load("results/sr_data_controls/expectations.npz") as source:
        expectations = {int(key): source[key] for key in source.files}
        expectation = expectations[100]
    frame = pd.read_csv("data/processed/sr_expanded/windows.csv")
    raw = np.load("data/processed/sr_expanded/raw_counts.npy", mmap_mode="r")
    assert raw.shape == (len(frame), 1024, 1024)
    val_rows = pd.read_csv("data/processed/sr_repeat_target/windows.csv")
    val_target = np.load("data/processed/sr_repeat_target/raw_counts.npy", mmap_mode="r")
    validation = []
    fallback = []
    for index, row in val_rows[val_rows.split == "val"].iterrows():
        counts1 = np.asarray(raw[int(row.original_index)], dtype="float32")
        counts2 = np.asarray(val_target[index], dtype="float32")
        high1, low1, predictions = infer(counts1, models, scale)
        high2 = torch.from_numpy(counts2[None, None])
        block2 = F.avg_pool2d(high2, 4)
        fallback.append(float(((block2 == 0) & (low1 > 0)).sum() / (low1 > 0).sum()))
        normalized_target = project_counts(high2, low1)
        for method in METHODS:
            validation.append(dict(start=int(row.start), group=int(row.group), method=method,
                                   objective=common_objective(predictions[method], normalized_target,
                                                              scale, expectations),
                                   repeat1_allocation_nll=float(conditional_allocation_nll(predictions[method], high1, low1)),
                                   repeat2_allocation_nll=float(conditional_allocation_nll(predictions[method], high2, low1))))
    pd.DataFrame(validation).to_csv(OUT / "validation_comparison.csv", index=False)
    pd.DataFrame(validation).groupby("method")[["objective", "repeat1_allocation_nll", "repeat2_allocation_nll"]].mean().to_csv(
        OUT / "validation_summary.csv"
    )
    rows = []
    for (start, end), group in frame[frame.split == "test"].groupby(["start", "end"]):
        assert len(group) == 2 and set(group.replicate) == {1, 2}
        pair = {}
        for repeat in (1, 2):
            idx = int(group.index[group.replicate == repeat][0])
            row = frame.iloc[idx]
            counts = np.asarray(raw[idx], dtype="float32") * float(row.depth_multiplier)
            high, low, predictions = infer(counts, models, scale)
            pair[repeat] = (high, low, predictions)
            for method in METHODS:
                rows.append(dict(split="same_repeat", start=int(start), group=int(row.group),
                                 repeat=repeat, method=method,
                                 **metrics(predictions[method], high, scale, expectation)))
        high1, low1, predictions = pair[1]
        high2 = pair[2][0]
        target_projected = project_counts(high2, low1)
        for method in METHODS:
            pred = predictions[method]
            rows.append(dict(split="cross_repeat_native", start=int(start),
                             group=int(group.group.iloc[0]), repeat=1, method=method,
                             **metrics(pred, high2, scale, expectation)))
            rows.append(dict(split="cross_repeat_projected", start=int(start),
                             group=int(group.group.iloc[0]), repeat=1, method=method,
                             **metrics(pred, target_projected, scale, expectation)))
        print("SR paired test", int(start), flush=True)
    result = pd.DataFrame(rows)
    assert result.groupby(["split", "method"]).size().to_dict() == {
        (split, method): 32 if split == "same_repeat" else 16
        for split in ("same_repeat", "cross_repeat_native", "cross_repeat_projected")
        for method in METHODS
    }
    result.to_csv(OUT / "test_per_window.csv", index=False)
    result.groupby(["split", "method"])[["psnr", "ssim", "oe_100bp_masked_ssim"]].mean().to_csv(
        OUT / "test_summary.csv"
    )
    intervals = {}
    primary = "conditional_rep2" if args.conditional else "cross_repeat_target"
    comparators = ("projected_bicubic", "sparse_matched", "dense_multiscale")
    if args.conditional:
        comparators += ("cross_repeat_target", "conditional_rep1")
    for split in ("same_repeat", "cross_repeat_native", "cross_repeat_projected"):
        subset = result[result.split == split]
        if split == "same_repeat":
            subset = subset[subset.repeat == 1]
        for metric in ("psnr", "ssim", "oe_100bp_masked_ssim"):
            for comparator in comparators:
                intervals[f"{split}_{metric}_versus_{comparator}"] = group_interval(
                    subset, metric, primary, comparator
                )
    save_json(OUT / "paired_uncertainty.json", intervals)
    save_json(OUT / "evaluation_protocol.json", dict(
        source="Frozen original WT repeat1 and repeat2 test windows, 8 spatial 100kb blocks",
        methods=METHODS, n_same_repeat=32, n_cross_repeat=16,
        primary_hypothesis=primary,
        cross_repeat_native="Compare source repeat1 40bp prediction to depth-normalized repeat2 native10bp",
        cross_repeat_projected="Compare to repeat2 native detail projected onto source repeat1 measured40bp totals",
        measure="Native clipped log PSNR/SSIM and masked 100bp log O/E SSIM, 10.24kb patches",
        count_conservation="Exact by construction for all methods",
        validation_zero_target_fallback_mean=float(np.mean(fallback)),
        validation_comparison="Same 12 validation loci, repeat2 native allocation projected to repeat1 measured40bp totals",
        comparison="Fixed seed42; 2000 100kb-block bootstrap draws; all test positions previously observed",
        limitations=["Two biological repeats only", "Test results retrospective, no new hyperparameter selection",
                     "Projected cross-repeat target shares the source 40bp total by construction"],
    ))
    print(result.groupby(["split", "method"])[["psnr", "ssim", "oe_100bp_masked_ssim"]].mean())


if __name__ == "__main__":
    main()

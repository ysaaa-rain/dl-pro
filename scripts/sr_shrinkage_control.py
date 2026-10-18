"""Validation-selected, count-preserving shrinkage for the fixed SR model.

All candidates share the same measured 40 bp block totals. The held-out
spatial test blocks are evaluated only after a validation selection is saved.
The test locations were observed in prior experiments, so this remains a
retrospective comparison.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from dlpro.refinement import ContextCountSR, project_counts
from dlpro.structure_losses import structure_terms
from dlpro.superresolution import scores


ROOT = Path("data/processed/sr_expanded")
OUT = Path("results/sr_shrinkage_control")
RECIPES = (
    "projected_bicubic", "dense_multiscale_seed42",
    "blend_025", "blend_050", "blend_075",
    "adaptive_k1", "adaptive_k4", "adaptive_k16",
)


def predictions(counts, model, scale):
    high = torch.from_numpy(counts[None, None].astype("float32"))
    low = F.avg_pool2d(high, 4)
    base = project_counts(
        F.interpolate(low, scale_factor=4, mode="bicubic", align_corners=False), low
    )
    with torch.no_grad():
        learned = torch.expm1(model(torch.log1p(low) / scale) * scale)
    result = {"projected_bicubic": base, "dense_multiscale_seed42": learned}
    for alpha, name in ((0.25, "blend_025"), (0.5, "blend_050"), (0.75, "blend_075")):
        result[name] = base + alpha * (learned - base)
    for k in (1, 4, 16):
        total = 16 * low
        weight = F.interpolate(total / (total + k), scale_factor=4, mode="nearest")
        result[f"adaptive_k{k}"] = base + weight * (learned - base)
    for name, pred in result.items():
        assert torch.isfinite(pred).all() and pred.min() >= 0, name
        error = (F.avg_pool2d(pred, 4) - low).abs().max().item()
        assert error < 1e-4, (name, error)
    return high, result


def objective(pred, target, scale, expectations):
    native = F.mse_loss(
        (torch.log1p(pred) / scale).clamp(0, 1),
        (torch.log1p(target) / scale).clamp(0, 1),
    )
    terms = structure_terms(pred, target, expectations)
    mse = torch.stack([pair[0] for pair in terms.values()]).mean()
    ssim_loss = torch.stack([pair[1] for pair in terms.values()]).mean()
    return float((native + 0.1 * mse + 0.05 * ssim_loss).item())


def test_metrics(pred, target, scale, expectation_100):
    pred_log = (torch.log1p(pred) / scale).clamp(0, 1).numpy()[0, 0]
    target_log = (torch.log1p(target) / scale).clamp(0, 1).numpy()[0, 0]
    native = scores(target_log, pred_log)
    shape_loss = structure_terms(pred, target, {100: expectation_100})[100][1]
    return dict(**native, oe_100bp_masked_ssim=float(1 - shape_loss.item()))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    frame = pd.read_csv(ROOT / "windows.csv")
    raw = np.load(ROOT / "raw_counts.npy", mmap_mode="r")
    assert raw.shape == (len(frame), 1024, 1024)
    checkpoint = torch.load(
        "results/sr_data_controls/dense_multiscale_seed42.pt", weights_only=True
    )
    assert checkpoint["recipe"] == "dense_multiscale"
    scale = float(checkpoint["scale"])
    model = ContextCountSR(scale).eval()
    model.load_state_dict(checkpoint["state_dict"])
    with np.load("results/sr_data_controls/expectations.npz") as source:
        expectations = {int(key): source[key] for key in source.files}
    assert set(expectations) == {50, 100, 160}

    validation = []
    for i, row in frame[frame.split == "val"].iterrows():
        high, variants = predictions(
            np.asarray(raw[i], dtype="float32") * float(row.depth_multiplier), model, scale
        )
        for name in RECIPES:
            validation.append(
                dict(index=int(i), group=int(row.group), recipe=name,
                     objective=objective(variants[name], high, scale, expectations))
            )
    pd.DataFrame(validation).to_csv(OUT / "validation_candidates.csv", index=False)
    means = pd.DataFrame(validation).groupby("recipe").objective.mean()
    selected = min(RECIPES, key=lambda name: (means[name], RECIPES.index(name)))
    (OUT / "selection.json").write_text(
        json.dumps(
            dict(selected=selected, validation_objective=means.to_dict(),
                 selection_only_validation=True, checkpoint=str(checkpoint["recipe"]),
                 recipes=RECIPES, test_evaluated=False), indent=2
        ), encoding="utf-8"
    )
    print("validation selected", selected, means.to_dict(), flush=True)

    comparators = tuple(dict.fromkeys(("projected_bicubic", "dense_multiscale_seed42", selected)))
    test = []
    for i, row in frame[frame.split == "test"].iterrows():
        high, variants = predictions(
            np.asarray(raw[i], dtype="float32") * float(row.depth_multiplier), model, scale
        )
        for name in comparators:
            test.append(
                dict(index=int(i), group=int(row.group), start=int(row.start),
                     replicate=int(row.replicate), recipe=name,
                     **test_metrics(variants[name], high, scale, expectations[100]))
            )
        print("test", i, flush=True)
    pd.DataFrame(test).to_csv(OUT / "test_windows.csv", index=False)
    summary = pd.DataFrame(test).groupby("recipe")[["psnr", "ssim", "oe_100bp_masked_ssim"]].mean()
    summary.to_csv(OUT / "test_summary.csv")

    transfer = []
    for (start, _), group in frame[frame.split == "test"].groupby(["start", "end"]):
        one = group[group.replicate == 1]
        two = group[group.replicate == 2]
        assert len(one) == len(two) == 1
        i, j = int(one.index[0]), int(two.index[0])
        _, variants = predictions(np.asarray(raw[i], dtype="float32"), model, scale)
        target = torch.from_numpy(
            (np.asarray(raw[j], dtype="float32") * float(frame.loc[j, "depth_multiplier"]))[None, None]
        )
        source_high = torch.from_numpy(np.asarray(raw[i], dtype="float32")[None, None])
        transfer.append(
            dict(start=int(start), group=int(frame.loc[i, "group"]),
                 recipe="source_repeat1_10bp",
                 **test_metrics(source_high, target, scale, expectations[100]))
        )
        for name in comparators:
            transfer.append(
                dict(start=int(start), group=int(frame.loc[i, "group"]), recipe=name,
                     **test_metrics(variants[name], target, scale, expectations[100]))
            )
    pd.DataFrame(transfer).to_csv(OUT / "cross_repeat_windows.csv", index=False)
    pd.DataFrame(transfer).groupby("recipe")[["psnr", "ssim", "oe_100bp_masked_ssim"]].mean().to_csv(
        OUT / "cross_repeat_summary.csv"
    )
    (OUT / "protocol.json").write_text(
        json.dumps(
            dict(
                purpose="Count-preserving shrinkage of frozen model toward projected bicubic",
                selection="Original 12 WT repeat1 validation windows, common native plus multiscale O/E objective",
                test="Original 32 spatial test windows; no test-based reselection",
                cross_repeat="Input WT repeat1 40bp, compare to WT repeat2 10bp at 16 matching test loci",
                shape_metric="Masked 100bp O/E SSIM on 10.24kb windows, distinct from prior 25.6kb known-context metric",
                limitations=["Previously observed test positions; retrospective result",
                             "Only two biological repeats and eight test spatial blocks",
                             "Within-repeat target shares reads with low-resolution input"],
            ), indent=2
        ), encoding="utf-8"
    )
    print(summary.to_string(), flush=True)


if __name__ == "__main__":
    main()

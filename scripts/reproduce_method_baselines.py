"""Fresh historical inference, not a claim of retraining historical models."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from dlpro.coverage_support import coarse_support
from dlpro.experiment_core import LENGTH, SubpixelSR
from dlpro.method_review import DEFAULT_CONFIG, load_config, restore_git_asset, sha256, write_json
from dlpro.refinement import morphology_features, project_counts
from dlpro.superresolution import scores
from discovery_dense_cv import select


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()
    cfg = load_config(args.config)
    out = Path(cfg["output"]) / "baseline_reproduced"
    out.mkdir(parents=True, exist_ok=True)
    # Each subprocess exit code is captured by the enclosing recorded stage.
    for command in (
        ["scripts/audit_task2_task5_iteration.py", "--run-tests", "--output", str(out / "audit")],
        ["scripts/evaluate_sr_repeat_target.py", "--conditional", "--output", str(out / "sr")],
    ):
        subprocess.run([sys.executable, *command], check=True)
    fresh = pd.read_csv(out / "sr/test_per_window.csv")
    old = pd.read_csv("results/sr_conditional/test_per_window.csv")
    keys = ["split", "start", "group", "repeat", "method"]
    paired = fresh.merge(old, on=keys, suffixes=("_fresh", "_old"), validate="one_to_one")
    metric_names = ["psnr", "ssim", "oe_100bp_masked_ssim"]
    delta = {m: float(abs(paired[m + "_fresh"] - paired[m + "_old"]).max()) for m in metric_names}
    assert len(paired) == len(fresh) == len(old) and all(v < 1e-6 for v in delta.values())
    known = pd.read_csv("data/processed/known_windows.csv")
    scan = pd.read_csv("data/processed/scan_windows.csv")
    folds = pd.read_csv("results/classification_cv/out_of_fold_predictions.csv").set_index("structure_id").loc[known.structure_id, "fold"].to_numpy()
    calibration = json.loads(Path("results/discovery_support_cv/protocol.json").read_text(encoding="utf-8"))
    centers = scan.center.to_numpy()
    distance = abs(centers[:, None] - known.center.to_numpy()[None])
    hit = np.minimum(distance, LENGTH-distance) <= cfg["discovery"]["hit_radius_bp"]
    raw = {r: np.load(f"data/processed/scan_rep{r}.npz")["raw"] for r in (1, 2)}
    support = {r: coarse_support(np.load(f"data/processed/coverage_rep{r}.npz")["coverage"], centers) >= cfg["discovery"]["support_fraction"] for r in (1, 2)}
    rows = []
    for fold in range(cfg["discovery"]["folds"]):
        expected = np.load(f"results/discovery_support_cv/fold{fold}_parameters.npz")["expected"]
        protocol = calibration["fold_protocols"][fold]
        for repeat in (1, 2):
            values = morphology_features(raw[repeat], expected, support[repeat])
            mask = support[repeat][:, 0, :, None] & support[repeat][:, 0, None, :]
            density = np.log(np.maximum((raw[repeat][:, 0] * mask).sum((1, 2)) / mask.sum((1, 2)).clip(1), 1e-12))
            standardized = np.empty_like(values)
            for j, p in enumerate(protocol["calibration"]):
                standardized[:, j] = (np.log(values[:, j].clip(1e-6)) - p["slope"] * density - p["intercept"] - p["location"]) / p["spread"]
            score = {"maximum": standardized.max(1), "mean": standardized.mean(1), "positive_mean": np.maximum(standardized, 0).mean(1)}[protocol["choice"]]
            for budget in cfg["discovery"]["global_budgets"]:
                selected = select(score, centers, budget)
                recalled = hit[selected].any(0)
                rows.extend(dict(structure_id=known.structure_id.iloc[i], type=known.type.iloc[i], group=int(known.group.iloc[i]),
                                 fold=fold, repeat=repeat, budget=budget, recalled=bool(recalled[i])) for i in np.flatnonzero(folds == fold))
    recall = pd.DataFrame(rows)
    recall.to_csv(out / "discovery_oof.csv", index=False)
    recall.groupby(["budget", "repeat", "type"]).recalled.agg(["mean", "size"]).to_csv(out / "discovery_summary.csv")
    original = pd.read_csv("results/discovery_directional_cv/out_of_fold_recall.csv").query('recipe == "old_morphology"')
    compare = recall.merge(original, on=["structure_id", "fold", "repeat", "budget"], suffixes=("_fresh", "_old"), validate="one_to_one")
    assert len(compare) == len(recall) and compare.recalled_fresh.eq(compare.recalled_old).all()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    restored = [restore_git_asset(path, revision) for path in (
        "results/sr_expanded/model_seed42.pt", "results/sr_refined/projection_ablation_native.csv")]
    checkpoint = torch.load(restored[0]["path"], weights_only=True)
    scale = float(checkpoint["scale"])
    model = SubpixelSR().eval()
    model.load_state_dict(checkpoint["state_dict"])
    torch.set_num_threads(cfg["threads"])
    plan = pd.read_csv("data/processed/sr_expanded/windows.csv")
    native = np.load("data/processed/sr_expanded/raw_counts.npy", mmap_mode="r")
    rows = []
    with torch.no_grad():
        for index, row in plan[plan.split == "test"].iterrows():
            high = torch.from_numpy((native[index].astype("float32") * row.depth_multiplier)[None, None])
            low = F.avg_pool2d(high, cfg["sr"]["factor"])
            prediction = model((torch.log1p(low) / scale).clamp(0, 1)).clamp(0, 1)
            counts = project_counts(torch.expm1(prediction * scale), low)
            rows.append(dict(sample_index=int(index), replicate=int(row.replicate), method="projected_first_round_cnn",
                             **scores((torch.log1p(high) / scale).clamp(0, 1).numpy()[0, 0], (torch.log1p(counts) / scale).numpy()[0, 0])))
    formal = pd.DataFrame(rows)
    formal.to_csv(out / "formal_sr_native.csv", index=False)
    reference = pd.read_csv(restored[1]["path"]).query('method == "projected_first_round_cnn"')
    paired = formal.merge(reference, on=["sample_index", "replicate", "method"], suffixes=("_fresh", "_old"), validate="one_to_one")
    formal_delta = {m: float(abs(paired[m + "_fresh"] - paired[m + "_old"]).max()) for m in ("psnr", "ssim")}
    assert len(paired) == len(formal) and all(v < 1e-6 for v in formal_delta.values())
    register = [dict(task="task2", method="support_aware_morphology", input="WT100/200bp", target="No fitted test labels; fixed fold calibration",
                     split="Original344 labels/56 groups/5fold", training="Reuse frozen train-only background parameters; recompute morphology from cached raw matrices",
                     device="cpu", seed=42, status="inference_reproduced_not_refitted", output=str(out / "discovery_oof.csv"))]
    for method, part in fresh.groupby("method"):
        register.append(dict(task="task5", method=method, input="40bp mean from native10bp WT count image",
                             split="Original8 test100kb blocks;32same/16cross", device="cpu", seed=42,
                             training="Reuse historical validation-selected checkpoint; no historical training repeated",
                             status="checkpoint_inference_reproduced" if method != "projected_bicubic" else "algorithm_reproduced",
                             output=str(out / "sr/test_per_window.csv")))
    register.append(dict(task="task5", method="projected_first_round_cnn", split="Original32same-repeat test windows", seed=42,
                         status="formal_checkpoint_inference_reproduced", device="cpu", output=str(out / "formal_sr_native.csv")))
    write_json(out / "baseline_register.json", register)
    write_json(out / "reproduction_checks.json", dict(discovery_binary_outcomes_match=True, sr_max_abs_delta=delta,
                                                      formal_sr_max_abs_delta=formal_delta, restored_assets=restored,
                                                      config_sha256=sha256(args.config, True), historical_training_rerun=False))
    print("reproduced", recall.groupby(["budget", "repeat"]).recalled.mean().to_dict(), delta, formal_delta, flush=True)


if __name__ == "__main__":
    main()

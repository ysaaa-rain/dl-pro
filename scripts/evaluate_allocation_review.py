"""One frozen SR evaluator for native, structural, count and spatial comparisons."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from dlpro.method_review import DEFAULT_CONFIG, load_config, sha256, write_json
from dlpro.refinement import project_counts
from dlpro.sr_review import infer_model, load_data, make_batch, make_model
from dlpro.structure_losses import structure_terms
from dlpro.superresolution import scores


NUMERIC_FLOOR = 1e-8
COUNT_RELATIVE_TOLERANCE = 1e-5


def simple_predictions(low, profile, factor):
    n = low.shape[-1] * factor
    d = abs(torch.arange(n)[:, None] - torch.arange(n)[None])
    prior = torch.as_tensor(profile, dtype=low.dtype)[d][None, None]
    result = {"uniform": F.interpolate(low, scale_factor=factor, mode="nearest"),
              "train_distance_prior": project_counts(prior, low)}
    for mode in ("bilinear", "bicubic"):
        raw = F.interpolate(low, scale_factor=factor, mode=mode, align_corners=False).clamp_min(0)
        result[mode + "_unprojected"] = raw
        result["projected_" + mode] = project_counts(raw, low)
    return result


def predict_all(models, high, profile, scale, sr):
    low = F.avg_pool2d(high, sr["factor"])
    predictions = {(name, -1): pred for name, pred in simple_predictions(low, profile, sr["factor"]).items()}
    for (arm, seed), model in models.items():
        predictions[(arm, seed)] = infer_model(model, arm, low, scale, sr)[0]
    return low, predictions


def metrics(pred, target, low, bicubic, scale, expectations, sr):
    pa, ta = pred.numpy()[0, 0], target.numpy()[0, 0]
    native = scores((torch.log1p(target) / scale).clamp(0, 1).numpy()[0, 0],
                    (torch.log1p(pred) / scale).clamp(0, 1).numpy()[0, 0])
    output = dict(psnr=native["psnr"], ssim=native["ssim"], log_mse=native["mse"])
    ix = np.arange(len(pa))
    distance = (ix[None, :] - ix[:, None]) * sr["native_bp"]
    unique = distance >= 0
    error = (pa - ta)[unique]
    y, p = ta[unique].astype(float), pa[unique].astype(float)
    output["raw_mae"] = float(abs(error).mean())
    output["raw_nrmse"] = float(np.sqrt(np.mean(error ** 2) / max(np.mean(y ** 2), NUMERIC_FLOOR)))
    yp = np.maximum(p, NUMERIC_FLOOR)
    poisson = yp - y
    nonzero = y > 0
    poisson[nonzero] += y[nonzero] * np.log(y[nonzero] / yp[nonzero])
    output["stabilized_poisson_deviance_per_count"] = float(2 * poisson.sum() / max(y.sum(), NUMERIC_FLOOR))
    for lower, upper in zip(sr["distance_bands_bp"][:-1], sr["distance_bands_bp"][1:]):
        mask = (distance >= lower) & (distance < upper)
        output[f"raw_mae_{lower}_{upper}bp"] = float(abs(pa[mask] - ta[mask]).mean())
    for bp, (_, loss) in structure_terms(pred, target, expectations).items():
        output[f"oe_{bp}bp_ssim"] = float(1 - loss)
    zero_source = F.interpolate((low == 0).float(), scale_factor=sr["factor"], mode="nearest").numpy()[0, 0].astype(bool)
    output["target_mass_in_zero_source_fraction"] = float(ta[zero_source & unique].sum() / max(ta[unique].sum(), NUMERIC_FLOOR))
    difference = pred - bicubic
    output["change_from_bicubic_relative_rms"] = float(torch.sqrt(difference.square().mean() / bicubic.square().mean().clamp_min(NUMERIC_FLOOR)))
    output["input_mean"] = float(low.mean())
    output["input_zero_block_fraction"] = float((low == 0).float().mean())
    relative = (F.avg_pool2d(pred, sr["factor"]) - low).abs() / low.clamp_min(NUMERIC_FLOOR)
    output["block_mean_max_relative_error"] = float(relative.max())
    output["symmetry_max_absolute_error"] = float((pred - pred.transpose(-1, -2)).abs().max())
    output["symmetry_max_relative_error"] = output["symmetry_max_absolute_error"] / max(float(pred.max()), NUMERIC_FLOOR)
    output["prediction_min"] = float(pred.min())
    output["prediction_max"] = float(pred.max())
    return output


def interval(delta, groups, cfg):
    groups, inverse = np.unique(groups, return_inverse=True)
    sums = np.bincount(inverse, weights=delta)
    counts = np.bincount(inverse)
    weights = np.random.default_rng(cfg["seed"]).multinomial(len(groups), np.full(len(groups), 1 / len(groups)), cfg["bootstrap_draws"])
    estimates = (weights @ sums) / (weights @ counts)
    lo, hi = np.quantile(estimates, [.025, .975])
    return dict(delta=float(np.mean(delta)), low=float(lo), high=float(hi), groups=len(groups))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()
    cfg = load_config(args.config)
    sr = cfg["sr"]
    root = Path(cfg["output"]) / sr.get("result_subdir", "sr")
    out = root / "evaluation"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "per_window.csv").exists():
        raise FileExistsError("Preserve the completed frozen evaluation")
    torch.set_num_threads(cfg["threads"])
    import json
    train_records = []
    for path in (Path(cfg["output"]) / "runs").glob("*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        command = record.get("command", [])
        if command and command[0].endswith("train_allocation_review.py") and record.get("config") == cfg and record.get("status") == "completed":
            train_records.append(record)
    assert len(train_records) == 1, "Require exactly one completed run with identical resolved configuration"
    for name in ("src/dlpro/allocation_model.py", "src/dlpro/sr_review.py", "src/dlpro/refinement.py", "scripts/train_allocation_review.py"):
        assert sha256(name, True) == train_records[0]["code_sha256"][name], f"Training code drift: {name}"
    plan, raw, depth = load_data(cfg)
    parameters = np.load(root / "normalization.npz")
    scale = float(parameters["scale"])
    profile = parameters["profile"]
    expectations = {bp: parameters[f"expected_{bp}"] for bp in sr["morphology_scales_bp"]}
    models, selected = {}, []
    for arm in sr["arms"]:
        for seed in sr["train_seeds"]:
            path = root / f"{arm}_seed{seed}.pt"
            checkpoint = torch.load(path, weights_only=True)
            assert checkpoint["config_sha256"] == sha256(args.config, True)
            history = pd.read_csv(root / f"{arm}_seed{seed}_history.csv")
            best = history.loc[history.validation_objective.idxmin()]
            assert checkpoint["selected_step"] == int(best.step)
            assert abs(checkpoint["validation_objective"] - best.validation_objective) < 1e-7
            assert set(plan.iloc[checkpoint["training_indices"]].split) == {"train"}
            model = make_model(arm, scale, profile, sr).eval()
            model.load_state_dict(checkpoint["state_dict"])
            models[(arm, seed)] = model
            selected.append(dict(arm=arm, seed=seed, step=checkpoint["selected_step"], sha256=sha256(path)))
    rows = []
    validation_rows = []
    with torch.no_grad():
        for index, row in plan[plan.split == "val"].iterrows():
            for repeat in (0, 1):
                high, _ = make_batch(raw, depth, [index], [repeat])
                other, _ = make_batch(raw, depth, [index], [1-repeat])
                low, predictions = predict_all(models, high, profile, scale, sr)
                bicubic = predictions[("projected_bicubic", -1)]
                for (method, seed), pred in predictions.items():
                    for comparison, target in (("same_repeat", high), ("cross_repeat", other)):
                        validation_rows.append(dict(group=int(row.group), start=int(row.start), source_repeat=repeat+1,
                                                    method=method, seed=seed, comparison=comparison,
                                                    constrained=not method.endswith("unprojected"),
                                                    **metrics(pred, target, low, bicubic, scale, expectations, sr)))
            print("validation selection metrics", int(row.start), flush=True)
    validation_frame = pd.DataFrame(validation_rows)
    validation_frame.to_csv(out / "validation_per_window.csv", index=False)
    val_summary = validation_frame.groupby(["method", "comparison"])[["psnr", "ssim", "oe_100bp_ssim"]].mean()
    val_summary.to_csv(out / "validation_summary.csv")
    reference = "legacy_sparse_mse"
    safety = []
    for method in sorted(validation_frame.method.unique()):
        eligible = not method.endswith("unprojected")
        for comparison in ("same_repeat", "cross_repeat"):
            current = val_summary.loc[(method, comparison)]
            baseline = val_summary.loc[(reference, comparison)]
            eligible &= current.psnr >= baseline.psnr - sr["max_psnr_drop_db"]
            eligible &= current.ssim >= baseline.ssim - sr["max_ssim_drop"]
        safety.append(dict(method=method, eligible=bool(eligible),
                           cross_oe_100bp_ssim=float(val_summary.loc[(method, "cross_repeat"), "oe_100bp_ssim"])))
    allowed = [entry for entry in safety if entry["eligible"]]
    selected_method = max(allowed, key=lambda entry: entry["cross_oe_100bp_ssim"])["method"]
    from datetime import datetime, timezone
    write_json(out / "validation_selection.json", dict(selected_method=selected_method,
                                                      selection_completed_utc=datetime.now(timezone.utc).isoformat(),
                                                      rule="Constrained; same/cross validation PSNR and SSIM safety versus legacy; maximize cross100bp OE SSIM; mean metrics over both seeds; alphabetical tie order",
                                                      alternatives=safety, test_performance_used=False,
                                                      selected_checkpoint_sha256={f"{item['arm']}_seed{item['seed']}": item["sha256"] for item in selected}))
    print("validation-selected method", selected_method, flush=True)
    write_json(out / "test_lock.json", dict(test_inference_started_utc=datetime.now(timezone.utc).isoformat(),
                                           validation_selection_sha256=sha256(out / "validation_selection.json"),
                                           checkpoint_sha256={f"{item['arm']}_seed{item['seed']}": item["sha256"] for item in selected}))
    evaluation = plan[plan.split.isin(["new_test", "historical_test"])]
    probes = set(evaluation[evaluation.split == "new_test"].groupby("group").head(1).sort_values("start").sample(n=2, random_state=cfg["seed"]).sample_index.tolist())
    with torch.no_grad():
        for index, row in evaluation.iterrows():
            for repeat in (0, 1):
                high, _ = make_batch(raw, depth, [index], [repeat])
                other, _ = make_batch(raw, depth, [index], [1-repeat])
                low, predictions = predict_all(models, high, profile, scale, sr)
                bicubic = predictions[("projected_bicubic", -1)]
                for (method, seed), pred in predictions.items():
                    constrained = not method.endswith("unprojected")
                    assert torch.isfinite(pred).all() and pred.min() >= 0
                    for comparison, target in (("same_repeat", high), ("cross_repeat", other)):
                        values = metrics(pred, target, low, bicubic, scale, expectations, sr)
                        if constrained:
                            assert values["block_mean_max_relative_error"] < COUNT_RELATIVE_TOLERANCE
                            assert values["symmetry_max_relative_error"] < COUNT_RELATIVE_TOLERANCE
                        rows.append(dict(split=row.split, group=int(row.group), start=int(row.start), source_repeat=repeat+1,
                                         method=method, seed=seed, comparison=comparison, constrained=constrained, **values))
                if index in probes and repeat == 0:
                    np.savez_compressed(out / f"probe_{int(row.start)}.npz", low=low.numpy()[0, 0], source=high.numpy()[0, 0],
                                        target=other.numpy()[0, 0], bicubic=bicubic.numpy()[0, 0],
                                        primary=predictions[(sr["primary_arm"], sr["train_seeds"][0])].numpy()[0, 0],
                                        prior=predictions[("train_distance_prior", -1)].numpy()[0, 0])
                print("evaluated", row.split, int(row.start), "source", repeat+1, flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(out / "per_window.csv", index=False)
    columns = ["psnr", "ssim", "raw_nrmse", "raw_mae", "stabilized_poisson_deviance_per_count",
               *[f"oe_{bp}bp_ssim" for bp in sr["morphology_scales_bp"]],
               *[f"raw_mae_{a}_{b}bp" for a,b in zip(sr["distance_bands_bp"][:-1], sr["distance_bands_bp"][1:])],
               "target_mass_in_zero_source_fraction", "change_from_bicubic_relative_rms"]
    result.groupby(["split", "comparison", "method", "seed"])[columns].mean().to_csv(out / "summary.csv")
    # Average seed metrics per physical position. This is not an ensemble prediction.
    mean = result.groupby(["split", "comparison", "method", "group", "start", "source_repeat"])[columns].mean().reset_index()
    intervals = []
    comparators = [sr["primary_comparator"], "legacy_sparse_mse", "projected_bicubic", "uniform", "context_sparse_shape", "allocation_expanded_cross"]
    for (split, comparison), subset in mean.groupby(["split", "comparison"]):
        for method in sorted(set(sr["arms"] + [selected_method])):
            for reference in comparators:
                if method == reference:
                    continue
                left = subset[subset.method == method]
                right = subset[subset.method == reference]
                paired = left.merge(right, on=["group", "start", "source_repeat"], suffixes=("_left", "_right"), validate="one_to_one")
                for metric in columns:
                    estimates = interval((paired[metric + "_left"] - paired[metric + "_right"]).to_numpy(), paired.group, cfg)
                    intervals.append(dict(split=split, comparison=comparison, method=method, reference=reference, metric=metric,
                                          aggregation="mean_of_seed_metrics", **estimates))
    pd.DataFrame(intervals).to_csv(out / "paired_intervals.csv", index=False)
    write_json(out / "protocol_and_checks.json", dict(selected_checkpoints=selected, output_count_semantics="Expected symmetric image counts; low is block mean",
                                                     model_count=len(models), test_locations=len(evaluation), rows=len(result),
                                                     numeric_floor=NUMERIC_FLOOR, relative_tolerance=COUNT_RELATIVE_TOLERANCE,
                                                     constrained_max_block_relative_error=float(result.loc[result.constrained, "block_mean_max_relative_error"].max()),
                                                     constrained_max_symmetry_relative_error=float(result.loc[result.constrained, "symmetry_max_relative_error"].max()),
                                                     all_nonnegative=bool(result.prediction_min.ge(0).all()),
                                                     scale=scale, same_and_cross_both_directions=True,
                                                     bootstrap="100kb spatial groups; mean of metrics over seeds, not an ensemble",
                                                     limitations=["Six new reserved groups within previously observed genome", "Raw targets remain noisy", "Stabilized deviance uses a numerical floor at impossible zero-source predictions", "Both repeats used at training locations; cross-repeat tests are spatially isolated, not a wholly unseen replicate"] ))


if __name__ == "__main__":
    main()

"""Verify actual model/data lineage, full coverage and frozen selection outcomes."""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

import numpy as np
import pandas as pd
import torch

from dlpro.experiment_core import ContactReader
from dlpro.method_review import DEFAULT_CONFIG, load_config, sha256, write_json
from dlpro.sr_review import load_data
from evaluate_allocation_review import simple_predictions


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = Path(cfg["output"])
    sr = cfg["sr"]
    checks = {}
    plan, raw, depth = load_data(cfg)
    assert plan.groupby("group").split.nunique().eq(1).all()
    assert (plan.start // cfg["block_bp"]).eq((plan.end-1) // cfg["block_bp"]).all()
    checks["native_inputs_spatially_disjoint_by_complete_block"] = True
    preparation = read_json(root / "data/preparation.json")
    for name, expected in preparation["cache_sha256"].items():
        assert sha256(Path(cfg["cache"]) / name) == expected
    sources = pd.read_csv("data/annotations/samples.csv").query('condition == "37C"').sort_values("replicate")
    probes = []
    for r, source in enumerate(sources.itertuples()):
        assert sha256(source.path) == source.sha256 == preparation["source_sha256"][str(r+1)]
        reader = ContactReader(source.path)
        for split in ("new_test", "historical_test"):
            index = int(plan.index[plan.split == split][0])
            direct = reader.window(int(plan.loc[index, "start"]) // sr["native_bp"], sr["patch_bins"], sr["native_bp"], False)
            assert np.array_equal(direct, raw[r][index])
            probes.append(dict(source_repeat=r+1, split=split, sample_index=index, exact_equal=True))
    checks["all_sources_cache_hashes_and_direct_test_windows_match"] = True
    protected = ["data/processed/known_windows.csv", "data/annotations/structures.csv", "data/annotations/samples.csv", "results/experiment_choices/choices.json"]
    for name in protected:
        expected = subprocess.check_output(["git", "rev-parse", "HEAD:" + name], text=True).strip()
        b = Path(name).read_bytes()
        assert hashlib.sha1(f"blob {len(b)}\0".encode()+b).hexdigest() == expected
    checks["frozen_annotations_original_splits_and_operational_choices_unchanged"] = True
    novelty = root / cfg["discovery"].get("result_subdir", "novelty")
    status = read_json(novelty / "status.json")
    assert status["complete_evaluation_coverage"] and not status["unassigned_tail_inputs_used"]
    known = pd.read_csv(root / "data/known_eligibility.csv")
    expected = set(known.loc[known.eligible & known.new_fold.ge(0), "structure_id"])
    frame = pd.read_csv(novelty / "point_scores.csv")
    for _, part in frame[frame.actual_type != "background"].groupby(["hidden_type", "repeat", "method"]):
        assert set(part.item_id) == expected and part.item_id.is_unique
    fits = read_json(novelty / "fits.json")
    checks["novelty_every_expected_case_scored_once_no_tail_default_training"] = True
    diagnostic = root / cfg.get("diagnostics_subdir", "diagnostics")
    fit_checks = read_json(diagnostic / "novelty_fitting_audit.json")
    assert len(fit_checks) == len(fits)
    assert all(not p["hidden_context_overlap"] and not p["held_fold_used_in_fit"] and p["all_fit_inputs_have_assigned_folds"] for p in fit_checks)
    checks["hidden_type_and_spatial_context_purged_from_all_fits"] = True
    model_root = root / sr.get("result_subdir", "sr")
    evaluation = model_root / "evaluation"
    lock, selection = read_json(evaluation / "test_lock.json"), read_json(evaluation / "validation_selection.json")
    assert datetime.fromisoformat(selection["selection_completed_utc"]) <= datetime.fromisoformat(lock["test_inference_started_utc"])
    assert sha256(evaluation / "validation_selection.json") == lock["validation_selection_sha256"]
    assert lock["checkpoint_sha256"] == selection["selected_checkpoint_sha256"]
    for arm in sr["arms"]:
        for seed in sr["train_seeds"]:
            name = f"{arm}_seed{seed}"
            assert sha256(model_root / (name + ".pt")) == lock["checkpoint_sha256"][name]
            cp = torch.load(model_root / (name + ".pt"), weights_only=True)
            history = pd.read_csv(model_root / (name + "_history.csv"))
            assert cp["selected_step"] == history.loc[history.validation_objective.idxmin(), "step"]
            assert set(plan.iloc[cp["training_indices"]].split) == {"train"}
    checks["all_models_and_validation_choice_locked_before_test_inference"] = True
    metrics = pd.read_csv(evaluation / "per_window.csv")
    profile = np.load(model_root / "normalization.npz")["profile"]
    baseline_names = simple_predictions(torch.ones(1, 1, 1, 1), profile, sr["factor"])
    method_keys = {(arm, seed) for arm in sr["arms"] for seed in sr["train_seeds"]}
    method_keys |= {(name, -1) for name in baseline_names}
    expected_keys = {(split, comparison, method, seed) for split in ("new_test", "historical_test")
                     for comparison in ("same_repeat", "cross_repeat") for method, seed in method_keys}
    assert set(metrics.groupby(["split", "comparison", "method", "seed"]).groups) == expected_keys
    for (split, comparison, method, seed), part in metrics.groupby(["split", "comparison", "method", "seed"]):
        expected_starts = set(plan.loc[plan.split == split, "start"])
        assert set(part.start) == expected_starts and len(part) == len(expected_starts) * len(raw)
        assert not part.duplicated(["start", "source_repeat"]).any()
    columns = ["psnr", "ssim", "raw_mae", "raw_nrmse", *[f"oe_{bp}bp_ssim" for bp in sr["morphology_scales_bp"]]]
    assert np.isfinite(metrics[columns]).all().all()
    checks["every_fixed_method_seed_repeat_and_location_evaluated_completely"] = True
    constrained = metrics[metrics.constrained]
    tolerance = read_json(evaluation / "protocol_and_checks.json")["relative_tolerance"]
    assert constrained.block_mean_max_relative_error.max() < tolerance
    assert constrained.symmetry_max_relative_error.max() < tolerance
    assert constrained.prediction_min.min() >= 0
    aligned_range = constrained.groupby(["split", "comparison", "start", "source_repeat"]).oe_160bp_ssim.agg(["min", "max"])
    assert (aligned_range["max"]-aligned_range["min"]).max() < tolerance
    checks["count_symmetry_nonnegativity_and_aligned160bp_control_hold"] = True
    archived, uncovered = [], []
    for path in (root / "runs").glob("*.json"):
        record = read_json(path)
        if "code_sha256" not in record:
            continue
        archive = Path(record.get("source_archive", root / "runs" / (path.stem + "_source.zip")))
        assert archive.exists()
        if "source_archive_sha256" in record:
            assert sha256(archive) == record["source_archive_sha256"]
        with zipfile.ZipFile(archive) as z:
            for name, expected_hash in record["code_sha256"].items():
                assert hashlib.sha256(z.read(name)).hexdigest() == expected_hash
        archived.append(path.stem)
    checks["recorded_run_source_archives_exactly_match_recorded_hashes"] = True
    for folder, archive_name, manifest_name in (
        (root / "legacy_audit_final", "test_sources.zip", "test_sources_sha256.json"),
        (root / "runs", "baseline_additional_source.zip", "baseline_additional_source.json"),
    ):
        with zipfile.ZipFile(folder / archive_name) as z:
            for name, expected_hash in read_json(folder / manifest_name).items():
                assert hashlib.sha256(z.read(name)).hexdigest() == expected_hash
    checks["baseline_subprocess_and_executed_test_sources_preserved"] = True
    legacy = read_json(root / "legacy_audit_final/audit.json")
    assert all(legacy["checks"].values())
    checks["historical_integrity_audit_and_tests_passed"] = True
    write_json(root / "final_audit.json", dict(checks=checks, direct_test_window_probes=probes,
                                               novel_case_count=len(expected), actual_novelty_fits=len(fits),
                                               constrained_max_block_error=float(constrained.block_mean_max_relative_error.max()),
                                               constrained_max_symmetry_relative_error=float(constrained.symmetry_max_relative_error.max()),
                                               aligned160bp_max_method_range=float((aligned_range["max"]-aligned_range["min"]).max()),
                                               archived_stages=archived, selected_method=selection["selected_method"],
                                               interpretation="Integrity of the recorded experiment, not proof of superior performance or biological novelty"))
    print(json.dumps(checks, indent=2), flush=True)


if __name__ == "__main__":
    main()

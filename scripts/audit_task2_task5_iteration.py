"""Verify frozen splits, outputs, source identity and SR constraints."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from dlpro.experiment_core import ContactReader, LENGTH
from dlpro.refinement import ContextCountSR
from dlpro.coverage_support import MINIMUM_COMMON_TRIANGLE_FRACTION, validate_candidate_table
from dlpro.method_review import load_config


OUT = Path("results/iteration_review")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        while part := source.read(8 * 1024 * 1024):
            h.update(part)
    return h.hexdigest()


def digest_code(path):
    """Hash canonical LF text so the audit is stable across Windows and Unix."""
    payload = Path(path).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(payload).hexdigest()


def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--run-tests", action="store_true")
    args = parser.parse_args()
    OUT = args.output
    OUT.mkdir(parents=True, exist_ok=True)
    checks = {}
    known = pd.read_csv("data/processed/known_windows.csv")
    folds = pd.read_csv("results/classification_cv/out_of_fold_predictions.csv")
    assert len(known) == 344 and known.structure_id.nunique() == 344
    joined = known[["structure_id", "group"]].merge(folds[["structure_id", "fold"]],
                                                   validate="one_to_one")
    assert joined.groupby("group").fold.nunique().eq(1).all() and joined.group.nunique() == 56
    checks["original344_labels_and56_spatial_groups_unchanged"] = True
    preparation = json.loads(Path("results/dense_scan/preparation.json").read_text())
    assert preparation["centers"] == 1451
    assert all(error == 0 for error in preparation["old_grid_max_abs_error"].values())
    checks["both_dense_scan_repeats_exactly_reproduce_original_grid"] = True
    discovery = ["discovery_dense_cv", "discovery_directional_cv", "discovery_type_weight_control"]
    for directory in discovery:
        frame = pd.read_csv(f"results/{directory}/out_of_fold_recall.csv")
        assert frame.groupby(["budget", "repeat", "recipe"]).structure_id.nunique().eq(344).all()
        assert frame.groupby(["budget", "repeat", "recipe"]).size().eq(344).all()
        mapped = frame.merge(folds[["structure_id", "fold"]], on="structure_id", suffixes=("", "_original"))
        assert mapped.fold.eq(mapped.fold_original).all()
    checks["discovery_oof_unique_and_original_fold_assignments_match"] = True
    plan = pd.read_csv("data/processed/sr_expanded/windows.csv")
    one = plan[plan.replicate == 1]
    assert one.split.value_counts().to_dict() == {"train": 64, "test": 16, "val": 12}
    assert plan.groupby("group").split.nunique().eq(1).all()
    test = plan[plan.split == "test"]
    assert test.group.nunique() == 8 and len(test) == 32
    assert test.groupby("start").replicate.nunique().eq(2).all()
    target_plan = pd.read_csv("data/processed/sr_repeat_target/windows.csv")
    assert len(target_plan) == 76
    assert set(target_plan.original_index) == set(one[one.split != "test"].index)
    assert not set(target_plan.group) & set(test.group)
    checks["sr64train12val_and8test_blocks_disjoint"] = True
    for directory, count in (("sr_repeat_target", 4), ("sr_conditional", 6)):
        metrics = pd.read_csv(f"results/{directory}/test_per_window.csv")
        assert metrics.method.nunique() == count
        assert metrics.groupby(["split", "method"]).size().to_dict() == {
            (split, method): 32 if split == "same_repeat" else 16
            for split in ("same_repeat", "cross_repeat_native", "cross_repeat_projected")
            for method in metrics.method.unique()
        }
        assert np.isfinite(metrics[["psnr", "ssim", "oe_100bp_masked_ssim"]]).all().all()
    checks["all_sr_fixed_arms_complete_finite_and_budget_matched"] = True
    candidate = pd.read_csv("results/discovery_directional_cv/candidate_context_audit.csv")
    candidate_protocol = load_config()["discovery"]
    independent = candidate[candidate.independent_context_member]
    assert not independent.known_full_25_6kb_context_overlap.any()
    assert independent.wt_oe_correlation_supported.ge(candidate_protocol["repeat_correlation_min"]).all()
    assert independent.supported_triangle_fraction.ge(MINIMUM_COMMON_TRIANGLE_FRACTION).all()
    centers = independent.center.to_numpy()
    distance = abs(centers[:, None] - centers[None])
    distance = np.minimum(distance, LENGTH-distance)
    assert (distance[np.triu_indices(len(centers), 1)] >= candidate_protocol["independent_spacing_bp"]).all()
    clusters = pd.read_csv("results/discovery_directional_cv/candidate_clusters.csv")
    candidate_counts = validate_candidate_table(candidate, clusters, candidate_protocol)
    checks["candidate_qc_context_independence_and_cluster_rules_consistent"] = True
    samples = pd.read_csv("data/annotations/samples.csv")
    sources = {}
    for row in samples[samples.condition.isin(["37C", "NoLig"])].itertuples():
        assert Path(row.path).stat().st_size == int(row.bytes)
        actual = digest(row.path)
        assert actual == row.sha256
        sources[row.sample_id] = actual
    checks["official_WT_and_NoLig_coolers_match_registered_SHA256"] = True
    native = np.load("data/processed/sr_expanded/raw_counts.npy", mmap_mode="r")
    covered = np.zeros(len(plan), bool)
    for path in sorted(Path("data/processed/sr_expanded").glob("counts_*.npz")):
        with np.load(path) as shard:
            first, values = int(shard["start"]), shard["raw"]
            assert not covered[first:first+len(values)].any()
            assert np.array_equal(native[first:first+len(values)], values)
            covered[first:first+len(values)] = True
    assert covered.all()
    checks["original_sr_array_exactly_matches_all_tracked_shards"] = True
    target_cache = np.load("data/processed/sr_repeat_target/raw_counts.npy", mmap_mode="r")
    repeat2_source = samples[(samples.condition == "37C") & (samples.replicate == 2)].iloc[0]
    reader = ContactReader(repeat2_source.path)
    target_probes = [int(target_plan[target_plan.split == split].index[0]) for split in ("train", "val")]
    for index in target_probes:
        extracted = reader.window(int(target_plan.loc[index, "start"]) // 10, 1024, 10, False)
        assert np.array_equal(extracted, target_cache[index])
    checks["train_and_validation_repeat2_targets_match_direct_Cooler_reads"] = True
    probes = [int(one[one.split == split].index[0]) for split in ("val", "test")]
    torch.set_num_threads(4)
    constraints = []
    for recipe in ("conditional_rep1", "conditional_rep2"):
        checkpoint = torch.load(f"results/sr_conditional/{recipe}_seed42.pt", weights_only=True)
        history = pd.read_csv(f"results/sr_conditional/{recipe}_history.csv")
        assert list(history.step) == list(range(80, 961, 80))
        best_row = history.loc[history.validation_nll.idxmin()]
        assert checkpoint["selected_step"] == int(best_row.step) and checkpoint["seed"] == 42
        assert abs(checkpoint["validation_objective"] - best_row.validation_nll) < 1e-7
        model = ContextCountSR(checkpoint["scale"]).eval()
        model.load_state_dict(checkpoint["state_dict"])
        for index in probes:
            high = torch.from_numpy(np.asarray(native[index], dtype="float32")[None, None])
            low = F.avg_pool2d(high, 4)
            with torch.no_grad():
                pred = torch.expm1(model(torch.log1p(low) / checkpoint["scale"]) * checkpoint["scale"])
            assert torch.isfinite(pred).all() and pred.min() >= 0
            relative = float(((F.avg_pool2d(pred, 4)-low).abs()/low.clamp_min(1e-3)).max())
            symmetry = float((pred-pred.transpose(-1, -2)).abs().max())
            assert relative < 1e-5 and symmetry < 1e-5
            constraints.append(dict(recipe=recipe, source_index=index,
                                    max_relative40bp_error=relative, symmetry_max_abs=symmetry))
    checks["conditional_checkpoints_are_validation_selected_and_count_consistent"] = True
    manifest = json.loads(Path("data/processed/fine_windows_sources.json").read_text())
    assert digest("data/processed/known_windows.csv") == manifest["known_plan_sha256"]
    checks["canonical_known_plan_bytes_match_original_signature"] = True
    if args.run_tests:
        result = subprocess.run([sys.executable, "-m", "pytest", "-q"], capture_output=True, text=True,
                                encoding="utf-8", errors="replace")
        test_lines = [line.strip() for line in (result.stdout + result.stderr).splitlines() if line.strip()]
        summary = test_lines[-1] if test_lines else f"pytest exit code: {result.returncode}"
        (OUT / "pytest.txt").write_text(summary + "\n", encoding="utf-8", newline="\n")
        assert result.returncode == 0, result.stdout + result.stderr
        checks["pytest_passed"] = True
    code_paths = [*Path("scripts").glob("*directional*.py"), *Path("scripts").glob("*conditional*.py"),
                  Path("scripts/audit_dense_candidates.py"), Path("scripts/evaluate_sr_repeat_target.py"),
                  Path("scripts/discovery_type_weight_control.py"), Path("scripts/audit_task2_task5_iteration.py"),
                  Path("src/dlpro/count_allocation.py"), Path("src/dlpro/ridge_features.py"),
                  Path("docs/第四轮固定实验协议.md")]
    code_paths += [Path("scripts") / name for name in (
        "train_sr_repeat_target.py", "prepare_sr_repeat_target.py", "prepare_dense_scan.py",
        "discovery_dense_cv.py", "discovery_fine_scale.py", "discovery_supervised_ranker.py",
        "discovery_repeat_transfer.py", "audit_sr_information_limit.py", "sr_shrinkage_control.py",
        "summarize_dense_discovery.py", "run_task2_task5_iteration.py", "restore_wt_sources.py",
        "restore_nolig_control.py", "plot_task2_task5_iteration.py")]
    audit = dict(
        checks=checks, baseline_git_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        source_sha256=sources, code_sha256={str(path).replace("\\", "/"): digest_code(path) for path in code_paths},
        count_constraint_probes=constraints,
        observed_candidate_counts=candidate_counts,
        model_sha256={str(path).replace("\\", "/"): digest(path) for path in
                      [Path("results/sr_repeat_target/model_seed42.pt"),
                       Path("results/sr_conditional/conditional_rep1_seed42.pt"),
                       Path("results/sr_conditional/conditional_rep2_seed42.pt")]},
        packages={name: importlib.metadata.version(name) for name in
                  ("numpy", "pandas", "torch", "scipy", "scikit-learn", "scikit-image", "cooler", "pytest")},
        python=platform.python_version(), platform=platform.platform(),
        limitations=["All test locations previously observed; retrospective same-study exploration",
                     "Group bootstrap intervals do not account for all accumulated model search",
                     "Known-structure detection is not evidence of a novel biological class"],
    )
    (OUT / "audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8", newline="\n")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()

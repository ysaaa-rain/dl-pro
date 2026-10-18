"""Reproduce the fixed October16-17 task-2/task-5 experiments by stage."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
STAGES = {
    "sources": [("restore_wt_sources.py",), ("restore_nolig_control.py",)],
    "prepare": [("prepare_dense_scan.py",), ("prepare_sr_repeat_target.py",)],
    "discovery": [
        ("discovery_supervised_ranker.py",), ("discovery_fine_scale.py",),
        ("discovery_repeat_transfer.py",), ("discovery_dense_cv.py",),
        ("summarize_dense_discovery.py",), ("audit_dense_candidates.py",),
        ("discovery_directional_cv.py",), ("summarize_directional_discovery.py",),
        ("audit_dense_candidates.py", "--results", "results/discovery_directional_cv",
         "--scan-prefix", "scan", "--recipe", "combined_validation_selected", "--id-prefix", "R"),
        ("discovery_type_weight_control.py",), ("audit_directional_nolig.py",),
    ],
    "superresolution": [
        ("audit_sr_information_limit.py",), ("sr_shrinkage_control.py",),
        ("train_sr_repeat_target.py",), ("evaluate_sr_repeat_target.py",),
        ("train_sr_conditional.py",), ("evaluate_sr_repeat_target.py", "--conditional"),
    ],
    "review": [("plot_task2_task5_iteration.py",), ("audit_task2_task5_iteration.py", "--run-tests")],
}


def restore_original_patch_cache():
    directory = ROOT / "data/processed/sr_expanded"
    plan = pd.read_csv(directory / "windows.csv")
    path = directory / "raw_counts.npy"
    if path.exists():
        assert np.load(path, mmap_mode="r").shape == (len(plan), 1024, 1024)
        return
    result = np.lib.format.open_memmap(path, mode="w+", dtype="uint32",
                                       shape=(len(plan), 1024, 1024))
    covered = np.zeros(len(plan), bool)
    for shard in sorted(directory.glob("counts_*.npz")):
        with np.load(shard) as source:
            start, counts = int(source["start"]), source["raw"]
            assert counts.shape[1:] == (1024, 1024) and not covered[start:start+len(counts)].any()
            result[start:start+len(counts)] = counts
            covered[start:start+len(counts)] = True
    assert covered.all(), "Missing tracked original count shards"
    result.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=[*STAGES, "all"], default="review")
    args = parser.parse_args()
    stages = STAGES if args.stage == "all" else {args.stage: STAGES[args.stage]}
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    environment["OMP_NUM_THREADS"] = "1"
    for stage, commands in stages.items():
        if stage in ("prepare", "superresolution"):
            restore_original_patch_cache()
        for command in commands:
            print(stage, " ".join(command), flush=True)
            subprocess.run([sys.executable, str(ROOT / "scripts" / command[0]), *command[1:]],
                           cwd=ROOT, env=environment, check=True)


if __name__ == "__main__":
    main()

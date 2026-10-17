"""Low-cost data and split diagnostics; do not inspect new SR test performance."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from torch.nn import functional as F
import torch

from dlpro.experiment_core import ContactReader, LENGTH
from dlpro.method_review import DEFAULT_CONFIG, load_config, sha256, write_json
from dlpro.sr_review import exact_count_log_quantile, load_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = Path(cfg["output"])
    out = root / cfg.get("diagnostics_subdir", "diagnostics")
    out.mkdir(parents=True, exist_ok=True)
    plan, raw, depth = load_data(cfg)
    sr = cfg["sr"]
    rows = []
    torch.set_num_threads(2)
    upper = np.triu(np.ones((sr["patch_bins"], sr["patch_bins"]), bool))
    block_n = sr["patch_bins"] // sr["factor"]
    block_upper = np.triu(np.ones((block_n, block_n), bool), 1)
    def blocks(a):
        f = sr["factor"]
        return a.reshape(block_n, f, block_n, f).transpose(0, 2, 1, 3)
    for index, row in plan[plan.split.isin(["train", "val"])].iterrows():
        pair = [a[index].astype('float64') for a in raw]
        for repeat in (0, 1):
            a, b = pair[repeat], pair[1-repeat]
            ba, bb = blocks(a), blocks(b)
            total_a, total_b = ba.sum((-2, -1)), bb.sum((-2, -1))
            zero = np.repeat(np.repeat(total_a == 0, sr["factor"], axis=0), sr["factor"], axis=1)
            keep = (total_a > 0) & (total_b > 0) & block_upper
            pa = ba[keep] / total_a[keep, None, None]
            pb = bb[keep] / total_b[keep, None, None]
            va, vb = (pa - 1 / sr["factor"] ** 2).ravel(), (pb - 1 / sr["factor"] ** 2).ravel()
            corr = np.dot(va, vb) / max(np.sqrt(np.dot(va, va) * np.dot(vb, vb)), 1e-12)
            rows.append(dict(sample_index=int(index), group=int(row.group), split=row.split, source_repeat=repeat+1,
                             native_zero_fraction=float((a == 0)[upper].mean()), input_zero_block_fraction=float((total_a == 0)[block_upper].mean()),
                             target_mass_in_zero_source_fraction=float(b[zero & upper].sum() / max(b[upper].sum(), 1)),
                             common_positive_offdiagonal_blocks=int(keep.sum()), normalized_within_block_correlation=float(corr)))
    diagnostics = pd.DataFrame(rows)
    diagnostics.to_csv(out / "sr_train_val_counts.csv", index=False)
    diagnostics.groupby(["split", "source_repeat"]).mean(numeric_only=True).to_csv(out / "sr_train_val_summary.csv")
    train = np.flatnonzero(plan.split == "train")
    exact_scale = exact_count_log_quantile(raw, train, depth, sr["normalization_quantile"])
    old_path = root / "sr/normalization.npz"
    write_json(out / "normalization_correction.json", dict(exact_training_pixel_log_quantile=exact_scale,
                                                          superseded_grid_scale=float(np.load(old_path)["scale"]) if old_path.exists() else None,
                                                          training_pixels=int(len(train) * len(raw) * sr["patch_bins"] ** 2),
                                                          test_pixels_used=0,
                                                          correction="Exact integer-count histograms, with separate per-repeat depth conversion"))
    known = pd.read_csv(root / "data/known_eligibility.csv")
    known["evaluation_eligible"] = known.eligible & known.new_fold.ge(0)
    known.to_csv(out / "known_evaluation_accounting.csv", index=False)
    other_overlap = []
    for row in known.itertuples():
        other = known[known.type != row.type]
        actual = ((other.start < row.end) & (other.end > row.start)).any()
        context = ((other.window_start < row.window_end) & (other.window_end > row.window_start)).any()
        other_overlap.append(dict(structure_id=row.structure_id, type=row.type, eligible=row.eligible,
                                  spatial_block=row.spatial_block, actual_other_type_overlap=bool(actual),
                                  full_context_other_type_overlap=bool(context)))
    pd.DataFrame(other_overlap).to_csv(out / "annotation_type_context_overlap.csv", index=False)
    geometry = []
    for split, part in plan.groupby("split"):
        geometry.append(dict(split=split, windows=len(part), groups=part.group.nunique(),
                             no_input_crosses_block=bool((part.start // cfg["block_bp"]).eq((part.end-1) // cfg["block_bp"]).all())))
    sources = pd.read_csv("data/annotations/samples.csv").query('condition == "37C"').sort_values("replicate")
    probes = []
    for r, source in enumerate(sources.itertuples()):
        reader = ContactReader(source.path)
        for split in ("train", "val"):
            index = int(plan.index[plan.split == split][0])
            direct = reader.window(int(plan.loc[index, "start"]) // sr["native_bp"], sr["patch_bins"], sr["native_bp"], False)
            equal = np.array_equal(direct, raw[r][index])
            assert equal
            probes.append(dict(split=split, sample_index=index, repeat=r+1, exact_direct_cooler_match=bool(equal)))
    # A train patch has 10240bp input; the context model uses at most 143 coarse
    # bins = 5720bp receptive field for the fixed six dilation blocks.
    receptive_coarse = 1 + 2 + sum(2*d + 2 for d in sr["dilations"]) + 2
    write_json(out / "split_and_geometry_audit.json", dict(split_counts=geometry, direct_cooler_probes=probes,
                                                         context_receptive_field_bp=receptive_coarse * sr["factor"] * sr["native_bp"],
                                                         whole_input_guard_bp=sr["patch_bins"] * sr["native_bp"],
                                                         raw_output_unit="expected symmetric image counts, not integer reads",
                                                         source_low_unit="mean of16 native image pixels; multiply by16 for image mass",
                                                         historical_test_kept=True, new_test_is_external=False))
    novelty_root = root / cfg["discovery"].get("result_subdir", "novelty")
    if (novelty_root / "fits.json").exists():
        scan = pd.read_csv(root / "data/scan_eligibility.csv")
        n = len(known)
        starts = np.r_[known.window_start, [int((c+100)//200)*200-12800 for c in scan.center]]
        ends = starts + cfg["discovery"]["max_context_bp"]
        folds = np.r_[known.new_fold, scan.new_fold]
        checks = []
        for fit in json.loads((novelty_root / "fits.json").read_text(encoding="utf-8")):
            ids = np.r_[fit["representation_training_indices"], fit["calibration_reference_indices"]]
            hidden = known[known.type == fit["hidden_type"]]
            overlap = ((starts[ids,None] < hidden.window_end.to_numpy()[None]) & (ends[ids,None] > hidden.window_start.to_numpy()[None])).any()
            train = np.asarray(fit["representation_training_indices"])
            wrong_fold = np.isin(folds[train], [fit["fold"], (fit["fold"]+1) % cfg["discovery"]["folds"]]).any()
            assigned = bool((folds[train] >= 0).all())
            assert not overlap and not wrong_fold and assigned
            checks.append(dict(hidden_type=fit["hidden_type"], fold=fit["fold"], hidden_context_overlap=bool(overlap), held_fold_used_in_fit=bool(wrong_fold), all_fit_inputs_have_assigned_folds=assigned))
        write_json(out / "novelty_fitting_audit.json", checks)
    print("data diagnostics complete; new SR test performance not accessed", flush=True)


if __name__ == "__main__":
    main()

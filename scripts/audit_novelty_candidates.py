"""Apply identical physical-context and repeat checks to every ranked candidate."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dlpro.coverage_support import coarse_support, MINIMUM_COMMON_TRIANGLE_FRACTION
from dlpro.experiment_core import LENGTH, hits
from dlpro.method_review import DEFAULT_CONFIG, load_config, nms, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()
    cfg = load_config(args.config)
    dc = cfg["discovery"]
    root = Path(cfg["output"])
    out = root / dc.get("result_subdir", "novelty")
    candidates = pd.read_csv(out / "selected_scan_windows.csv")
    known = pd.read_csv("data/processed/known_windows.csv")
    centers = pd.read_csv("data/processed/scan_windows.csv").center.to_numpy()
    raw = {r: np.load(f"data/processed/scan_rep{r}.npz")["raw"] for r in (1, 2)}
    support = {r: coarse_support(np.load(f"data/processed/coverage_rep{r}.npz")["coverage"], centers) >= dc["support_fraction"] for r in (1, 2)}
    fits = {(fit["hidden_type"], fit["fold"]): fit for fit in json.loads((out / "fits.json").read_text(encoding="utf-8"))}
    n = dc["input_bins"]
    distance = abs(np.arange(n)[:, None] - np.arange(n)[None])
    triangle = np.triu(np.ones((n, n), bool), 2)
    expectations, cached = {}, {}
    rows = []
    for row in candidates.itertuples():
        key = (row.hidden_type, row.fold, row.scan_index)
        if key not in cached:
            index = int(row.scan_index)
            fit_key = (row.hidden_type, row.fold)
            if fit_key not in expectations:
                ids = np.array(fits[fit_key]["background_training_indices"]) - len(known)
                assert (ids >= 0).all()
                valid = support[1][ids, 0, :, None] & support[1][ids, 0, None, :]
                expected = np.empty(n)
                for d in range(n):
                    a, m = raw[1][ids, 0][:, distance == d], valid[:, distance == d]
                    expected[d] = max(float(a[m].mean()) if m.any() else 0, 1e-8)
                expectations[fit_key] = expected[distance]
            expected = expectations[fit_key]
            valid_rows = support[1][index, 0] & support[2][index, 0]
            valid_rows &= (raw[1][index, 0].sum(-1) > 0) & (raw[2][index, 0].sum(-1) > 0)
            mask = triangle & valid_rows[:, None] & valid_rows[None]
            left, right = raw[1][index, 0] / expected, raw[2][index, 0] / expected
            fraction = mask.sum() / triangle.sum()
            correlation = np.corrcoef(left[mask], right[mask])[0, 1] if mask.sum() > 2 and left[mask].std() > 0 and right[mask].std() > 0 else np.nan
            strata = []
            for d in range(2, n):
                use = mask & (distance == d)
                if use.sum() > 2 and left[use].std() > 0 and right[use].std() > 0:
                    strata.append(np.corrcoef(left[use], right[use])[0, 1])
            context_known = bool(hits(int(row.center), dc["max_context_bp"], known, buffer=dc["known_buffer_bp"]).any())
            cached[key] = dict(common_triangle_fraction=float(fraction), wt_oe_pearson=float(correlation),
                               mean_distance_stratum_pearson=float(np.mean(strata)) if strata else None,
                               evaluable_distance_strata=len(strata), full_known_context_overlap=context_known,
                               passes_repeat_qc=bool(fraction >= MINIMUM_COMMON_TRIANGLE_FRACTION and np.isfinite(correlation) and correlation >= dc["repeat_correlation_min"]))
        rows.append(dict(hidden_type=row.hidden_type, fold=row.fold, repeat=row.repeat, method=row.method,
                         global_equivalent_budget=row.global_equivalent_budget, rank=row.rank, center=row.center,
                         score=row.score, **cached[key]))
    frame = pd.DataFrame(rows)
    frame["independent_unannotated"] = False
    summary = []
    keys = ["hidden_type", "repeat", "method", "global_equivalent_budget"]
    for values, part in frame.groupby(keys):
        eligible = part[part.passes_repeat_qc & ~part.full_known_context_overlap].sort_values("center")
        # Same genome-order greedy independence rule as the historical audit.
        accepted = []
        for index, row in eligible.iterrows():
            d = np.abs(eligible.loc[accepted, "center"].to_numpy() - row.center)
            if np.all(np.minimum(d, LENGTH-d) >= dc["independent_spacing_bp"]):
                accepted.append(index)
        frame.loc[accepted, "independent_unannotated"] = True
        summary.append(dict(zip(keys, values), selected=len(part), repeat_qc=int(part.passes_repeat_qc.sum()),
                            unannotated=int((~part.full_known_context_overlap).sum()),
                            repeat_qc_unannotated=len(eligible), independent=len(accepted),
                            enough_members_for_any_formal_cluster=len(accepted) >= dc["min_cluster_members"]))
    frame.to_csv(out / "candidate_quality_per_window.csv", index=False)
    pd.DataFrame(summary).to_csv(out / "candidate_quality_summary.csv", index=False)
    write_json(out / "candidate_quality_protocol.json", dict(minimum_common_triangle_fraction=MINIMUM_COMMON_TRIANGLE_FRACTION,
                                                              formal_source="results/discovery_directional_cv/candidate_audit_protocol.json",
                                                              independent_spacing_bp=dc["independent_spacing_bp"],
                                                              known_exclusion_width_bp=dc["max_context_bp"], known_buffer_bp=dc["known_buffer_bp"],
                                                              scope="Pseudo-held-type models on spatially eligible scan windows, not a selected full-genome production model",
                                                              joint_repeat_caveat="Joint method already uses both repeats; its correlation is descriptive, not independent validation",
                                                              cluster_interpretation="Enough members is only a size prerequisite; no candidate count alone establishes a new cluster"))


if __name__ == "__main__":
    main()

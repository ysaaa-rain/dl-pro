"""Training-background mismatch controls and non-forced candidate clustering.

This follows the completed pseudo-novelty comparison. It never changes scores,
thresholds, retained labels or previous evaluation outputs.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import HDBSCAN, KMeans
from threadpoolctl import threadpool_limits

from dlpro.coverage_support import coarse_support
from dlpro.experiment_core import LENGTH
from dlpro.method_review import DEFAULT_CONFIG, load_config, write_json


def representation(raw, rows, grid):
    """Per-image operator: no candidate-fitted PCA, scale or reference prototypes."""
    n = raw.shape[-1]
    distance = abs(np.arange(n)[:, None] - np.arange(n)[None])
    valid = (distance >= 2)[None] & rows[:, :, None] & rows[:, None, :]
    fields = []
    for channel in range(raw.shape[0]):
        profile = np.ones(n)
        for d in range(2, n):
            keep = (distance == d) & valid[channel]
            profile[d] = max(float(raw[channel][keep].mean()) if keep.any() else 0, 1e-8)
        field = np.where(valid[channel], np.log1p(raw[channel] / profile[distance]), 0)
        f = n // grid
        pooled = field.reshape(grid, f, grid, f).mean((1, 3))
        weights = valid[channel].reshape(grid, f, grid, f).mean((1, 3))
        values = (pooled / weights.clip(1e-8)).ravel()
        fields.extend(values / max(np.linalg.norm(values), 1e-8))
    return np.asarray(fields)


def correlations(a, b, row_mask):
    n = len(a)
    delta = abs(np.arange(n)[:, None] - np.arange(n)[None])
    mask = np.triu(np.ones((n, n), bool), 2) & row_mask[:, None] & row_mask[None]
    global_r = float(np.corrcoef(a[mask], b[mask])[0, 1]) if mask.sum() > 2 and a[mask].std() > 0 and b[mask].std() > 0 else np.nan
    strata = []
    for d in range(2, n):
        use = mask & (delta == d)
        if use.sum() > 2 and a[use].std() > 0 and b[use].std() > 0:
            strata.append(np.corrcoef(a[use], b[use])[0, 1])
    return global_r, float(np.mean(strata)) if strata else np.nan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()
    cfg = load_config(args.config)
    dc = cfg["discovery"]
    out = Path(cfg["output"]) / dc.get("result_subdir", "novelty")
    frame = pd.read_csv(out / "candidate_quality_per_window.csv")
    fits = json.loads((out / "fits.json").read_text(encoding="utf-8"))
    centers = pd.read_csv("data/processed/scan_windows.csv").center.to_numpy()
    known_n = len(pd.read_csv("data/processed/known_windows.csv"))
    raw = {r: np.load(f"data/processed/scan_rep{r}.npz")["raw"] for r in (1, 2)}
    support = {r: coarse_support(np.load(f"data/processed/coverage_rep{r}.npz")["coverage"], centers) >= dc["support_fraction"] for r in (1, 2)}
    indices = {int(c): i for i, c in enumerate(centers)}
    grid = dc["pooled_bins"]
    n = dc["input_bins"]
    delta = abs(np.arange(n)[:, None] - np.arange(n)[None])
    triangle = np.triu(np.ones((n, n), bool), 2)
    controls, cutoffs = [], {}
    for fit in fits:
        bg = np.asarray(fit["background_training_indices"]) - known_n
        nuisance = np.stack([np.log(raw[1][bg, 0].mean((1, 2)).clip(1e-12)),
                              (raw[1][bg, 0] == 0).mean((1, 2)), support[1][bg, 0].mean(1)], axis=1)
        standardized = (nuisance - nuisance.mean(0)) / nuisance.std(0).clip(1e-8)
        distances = cdist(standardized, standardized)
        groups = centers[bg] // cfg["block_bp"]
        distances[groups[:, None] == groups[None]] = np.inf
        expected = np.empty(n)
        valid = support[1][bg, 0, :, None] & support[1][bg, 0, None, :]
        for d in range(n):
            a, mask = raw[1][bg, 0][:, delta == d], valid[:, delta == d]
            expected[d] = max(float(a[mask].mean()) if mask.any() else 0, 1e-8)
        values = []
        for i in range(len(bg)):
            j = int(distances[i].argmin())
            if not np.isfinite(distances[i, j]):
                continue
            a, b = int(bg[i]), int(bg[j])
            rows = support[1][a, 0] & support[2][b, 0]
            global_r, stratum_r = correlations(raw[1][a, 0] / expected[delta], raw[2][b, 0] / expected[delta], rows)
            controls.append(dict(hidden_type=fit["hidden_type"], fold=fit["fold"], left_center=int(centers[a]), right_center=int(centers[b]),
                                 nuisance_distance=float(distances[i, j]), global_r=global_r, stratum_r=stratum_r))
            values.append((global_r, stratum_r))
        array = np.asarray(values)
        threshold = np.nanquantile(array, 1-dc["false_positive_rate"], axis=0)
        cutoffs[(fit["hidden_type"], fit["fold"])] = threshold
    pd.DataFrame(controls).to_csv(out / "training_mismatched_repeat_controls.csv", index=False)
    frame["above_global_training_null"] = [row.wt_oe_pearson > cutoffs[(row.hidden_type, row.fold)][0] for row in frame.itertuples()]
    frame["above_stratified_training_null"] = [row.mean_distance_stratum_pearson > cutoffs[(row.hidden_type, row.fold)][1] for row in frame.itertuples()]
    frame.to_csv(out / "candidate_matched_null_diagnostics.csv", index=False)
    shape_cache = {}
    for repeat in (1, 2):
        for center in frame.center.unique():
            index = indices[int(center)]
            shape_cache[(repeat, int(center))] = representation(raw[repeat][index], support[repeat][index], grid)
    keys = ["hidden_type", "repeat", "method", "global_equivalent_budget"]
    cluster_rows, memberships = [], []
    threadpool_limits(1)
    for values, part in frame.groupby(keys):
        # Grouping is exploratory/transductive. The representation is a fixed
        # per-image operator; no hidden labels or candidate-fitted transforms.
        part = part.reset_index(drop=True)
        vectors = {r: np.stack([shape_cache[(r, int(c))] for c in part.center]) for r in (1, 2)}
        labels = {}
        for algorithm in ("fixed_kmeans", "density_hdbscan"):
            for r in (1, 2):
                if algorithm == "fixed_kmeans":
                    labels[(algorithm, r)] = KMeans(n_clusters=min(dc["cluster_count"], len(part)), n_init=dc["kmeans_n_init"], random_state=cfg["seed"]).fit_predict(vectors[r])
                else:
                    labels[(algorithm, r)] = HDBSCAN(min_cluster_size=dc["min_cluster_members"], min_samples=dc["min_cluster_members"]).fit_predict(vectors[r])
            first, second = labels[(algorithm, 1)], labels[(algorithm, 2)]
            for label in np.unique(first):
                mask = first == label
                independent = mask & part.independent_unannotated.to_numpy()
                intersection, best_second = 0, -1
                for label2 in np.unique(second):
                    if label2 < 0:
                        continue
                    overlap = int((independent & (second == label2)).sum())
                    if overlap > intersection:
                        intersection, best_second = overlap, int(label2)
                known_first = int(part.loc[mask, "full_known_context_overlap"].sum())
                known_second = int(part.loc[second == best_second, "full_known_context_overlap"].sum()) if best_second >= 0 else 0
                minimum = dc["min_cluster_members"]
                provisional = bool(label >= 0 and independent.sum() >= minimum and known_first == 0
                                   and intersection >= minimum and known_second == 0)
                cluster_rows.append(dict(zip(keys, values), algorithm=algorithm, cluster=int(label), total_members=int(mask.sum()),
                                         independent_unannotated_members=int(independent.sum()), known_context_members=known_first,
                                         best_repeat2_cluster=best_second, independent_reproduced_members=intersection,
                                         provisional_shape_cluster=provisional,
                                         independent_members_above_stratified_null=int((independent & part.above_stratified_training_null.to_numpy()).sum())))
            for i, row in part.iterrows():
                memberships.append(dict(zip(keys, values), algorithm=algorithm, center=int(row.center),
                                        rep1_cluster=int(first[i]), rep2_cluster=int(second[i])))
    pd.DataFrame(cluster_rows).to_csv(out / "cluster_quality.csv", index=False)
    pd.DataFrame(memberships).to_csv(out / "cluster_membership.csv", index=False)
    write_json(out / "clustering_protocol.json", dict(descriptor=dc["cluster_descriptor"], minimum_cluster_members=dc["min_cluster_members"],
                                                     clustering="Exploratory transductive grouping only; no new biological class validation",
                                                     null="Train-background WT1 versus nearest nuisance-matched different100kb-block WT2;95th percentile descriptive cutoff",
                                                     multiple_testing="No genome-wide significance or FDR claim; correlated windows and post-selection QC prevent interpreting these ranks as p-values",
                                                     unchanged_formal_qc=True, score_models_refitted=False,
                                                     algorithms=["KMeans fixed6 control", "HDBSCAN formal minimum size and min_samples"],
                                                     provisional_shape_clusters=int(pd.DataFrame(cluster_rows).provisional_shape_cluster.sum())))


if __name__ == "__main__":
    main()

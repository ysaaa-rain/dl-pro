"""Leave-one-type-out novelty, with full-input spatial purging and fixed routes."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits

from dlpro.coverage_support import coarse_support
from dlpro.experiment_core import LENGTH, centered_indices, hits
from dlpro.method_review import DEFAULT_CONFIG, load_config, nms, spatial_cv_masks, write_json
from dlpro.novelty import empirical_rank, fit_routes, grouped_auc, shape_vectors
from dlpro.refinement import morphology_features


def overlap_inputs(starts, ends, intervals):
    if not len(intervals):
        return np.zeros(len(starts), bool)
    return ((starts[:, None] < intervals.window_end.to_numpy()[None])
            & (ends[:, None] > intervals.window_start.to_numpy()[None])).any(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()
    cfg = load_config(args.config)
    dc = cfg["discovery"]
    root = Path(cfg["output"])
    out = root / dc.get("result_subdir", "novelty")
    out.mkdir(parents=True, exist_ok=True)
    if (out / "point_scores.csv").exists():
        raise FileExistsError("Keep completed scores; use a new result subdirectory")
    known = pd.read_csv(root / "data/known_eligibility.csv")
    scan = pd.read_csv(root / "data/scan_eligibility.csv")
    nk = len(known)
    centers = np.r_[known.center, scan.center]
    starts = np.r_[known.window_start, [centered_indices(int(c))[1] * dc["input_scales_bp"][-1] for c in scan.center]]
    ends = starts + dc["max_context_bp"]
    groups = np.r_[known.spatial_block, scan.spatial_block]
    reference_groups = groups.copy()
    reference_groups[:nk] = known.group.to_numpy() + groups.max() + 1
    folds = np.r_[known.new_fold, scan.new_fold]
    eligible = np.r_[known.eligible, scan.eligible].astype(bool) & (folds >= 0)
    raw = {r: np.concatenate([np.load(f"data/processed/known_rep{r}.npz")["raw"],
                              np.load(f"data/processed/scan_rep{r}.npz")["raw"]]) for r in (1, 2)}
    support = {r: coarse_support(np.load(f"data/processed/coverage_rep{r}.npz")["coverage"], centers) >= dc["support_fraction"] for r in (1, 2)}
    annotated = np.array([hits(int(c), dc["max_context_bp"], known, buffer=dc["known_buffer_bp"]).any() for c in centers])
    is_scan = np.arange(len(centers)) >= nk
    known_type = np.r_[known.type.to_numpy(), np.full(len(scan), "background")]
    delta = abs(np.arange(dc["input_bins"])[:, None] - np.arange(dc["input_bins"])[None])
    records, recall, selections, fits, failures, confounds = [], [], [], [], [], []
    threadpool_limits(2)
    for hidden in sorted(known.type.unique()):
        hidden_rows = known[known.type == hidden]
        hidden_context = overlap_inputs(starts, ends, hidden_rows)
        for fold in range(dc["folds"]):
            val_fold = (fold + 1) % dc["folds"]
            training, val, test = spatial_cv_masks(eligible, folds, fold, dc["folds"])
            training &= ~hidden_context
            train_bg = np.flatnonzero(training & is_scan & ~annotated)
            references = np.flatnonzero(training & ~is_scan & (known_type != hidden))
            val_bg = np.flatnonzero(val & is_scan & ~annotated)
            val_refs = np.flatnonzero(val & ~is_scan & (known_type != hidden) & ~hidden_context)
            expected = np.empty((len(dc["input_scales_bp"]), dc["input_bins"]))
            try:
                if len(train_bg) < 2 or not len(references) or not len(val_bg):
                    raise ValueError("Insufficient fully purged background/reference/calibration data")
                for channel in range(len(expected)):
                    valid = support[1][train_bg, channel, :, None] & support[1][train_bg, channel, None, :]
                    for d in range(dc["input_bins"]):
                        a = raw[1][train_bg, channel][:, delta == d]
                        keep = valid[:, delta == d]
                        expected[channel, d] = max(float(a[keep].mean()) if keep.any() else 0, 1e-8)
                vectors, nuisance, morph = {}, {}, {}
                needed = np.unique(np.r_[train_bg, references, val_bg, val_refs, np.flatnonzero(test)])
                for r in (1, 2):
                    part_vectors, part_nuisance = shape_vectors(raw[r][needed], support[r][needed], expected, dc["pooled_bins"])
                    part_morph = morphology_features(raw[r][needed], expected, support[r][needed])
                    vectors[r] = np.zeros((len(centers), part_vectors.shape[1]), dtype=part_vectors.dtype)
                    nuisance[r] = np.zeros((len(centers), part_nuisance.shape[1]), dtype=part_nuisance.dtype)
                    morph[r] = np.zeros((len(centers), part_morph.shape[1]), dtype=part_morph.dtype)
                    vectors[r][needed], nuisance[r][needed], morph[r][needed] = part_vectors, part_nuisance, part_morph
                routes, diagnostic = fit_routes(vectors, nuisance, train_bg, references, val_bg, val_refs, reference_groups, dc)
                # Legacy morphology fitting is repeated here under the exact same spatial masks.
                standardized = {r: np.empty_like(morph[r]) for r in (1, 2)}
                for j in range(morph[1].shape[1]):
                    fit = HuberRegressor(max_iter=1000).fit(nuisance[1][train_bg, :1], np.log(morph[1][train_bg, j].clip(1e-6)))
                    residual = np.log(morph[1][train_bg, j].clip(1e-6)) - fit.predict(nuisance[1][train_bg, :1])
                    location = np.median(residual)
                    spread = max(1.4826 * np.median(abs(residual - location)), dc["scale_floor"])
                    for r in (1, 2):
                        standardized[r][:, j] = (np.log(morph[r][:, j].clip(1e-6)) - fit.predict(nuisance[r][:, :1]) - location) / spread
                aggregations = lambda a: dict(maximum=a.max(1), mean=a.mean(1), positive_mean=np.maximum(a, 0).mean(1))
                validation_scores = {}
                calibration_refs = val_refs if len(val_refs) else references
                for name, values in aggregations(standardized[1]).items():
                    y = np.r_[np.ones(len(calibration_refs)), np.zeros(len(val_bg))]
                    validation_scores[name] = roc_auc_score(y, np.r_[values[calibration_refs], values[val_bg]])
                chosen = max(validation_scores, key=validation_scores.get)
                base_calibration = aggregations(standardized[1])[chosen][val_bg]
                for r in (1, 2):
                    routes[r]["morphology_baseline"] = empirical_rank(aggregations(standardized[r])[chosen], base_calibration)
                assert not hidden_context[np.r_[train_bg, references, val_refs]].any()
                assert not np.isin(folds[np.r_[train_bg, references]], [fold, val_fold]).any()
                assert (folds[np.r_[train_bg, references, val_bg, val_refs]] >= 0).all()
                fits.append(dict(hidden_type=hidden, fold=fold, morphology_choice=chosen,
                                 validation_auc_allowed_vs_background=validation_scores, **diagnostic))
                test_points = np.flatnonzero(test & (~is_scan | ~annotated))
                test_scan = np.flatnonzero(test & is_scan)
                for r, methods in routes.items():
                    for method, score in methods.items():
                        threshold = float(np.quantile(score[val_bg], 1 - dc["false_positive_rate"]))
                        for index in test_points:
                            records.append(dict(hidden_type=hidden, fold=fold, repeat=r, method=method,
                                                item_id=(known.structure_id.iloc[index] if index < nk else f"scan_{index-nk}"),
                                                actual_type=known_type[index], group=int(groups[index]),
                                                score=float(score[index]), flagged=bool(score[index] > threshold),
                                                calibration_threshold=threshold))
                        for nuisance_name, values in zip(("log_density", "zero_fraction", "support"), nuisance[r][test_scan, :3].T):
                            value = float(spearmanr(score[test_scan], values).statistic)
                            confounds.append(dict(hidden_type=hidden, fold=fold, repeat=r, method=method,
                                                  nuisance=nuisance_name, spearman=None if not np.isfinite(value) else value))
                        for global_budget in dc["global_budgets"]:
                            budget = int(np.ceil(global_budget * len(test_scan) / len(scan)))
                            selected = test_scan[nms(score[test_scan], centers[test_scan], budget, dc["nms_bp"], LENGTH)]
                            test_known = np.flatnonzero(test & ~is_scan)
                            distance = abs(centers[selected, None] - centers[test_known][None])
                            recalled = (np.minimum(distance, LENGTH - distance) <= dc["hit_radius_bp"]).any(0)
                            for index, hit in zip(test_known, recalled):
                                recall.append(dict(hidden_type=hidden, fold=fold, repeat=r, method=method,
                                                   global_equivalent_budget=global_budget, actual_fold_budget=budget,
                                                   structure_id=known.structure_id.iloc[index], actual_type=known_type[index],
                                                   group=int(groups[index]), recalled=bool(hit)))
                            for rank, index in enumerate(selected, 1):
                                distance_all = abs(known.center.to_numpy() - centers[index])
                                hit = np.minimum(distance_all, LENGTH-distance_all) <= dc["hit_radius_bp"]
                                selections.append(dict(hidden_type=hidden, fold=fold, repeat=r, method=method,
                                                       global_equivalent_budget=global_budget, rank=rank,
                                                       scan_index=int(index-nk), center=int(centers[index]),
                                                       score=float(score[index]), hidden_hit=bool((hit & known.type.eq(hidden).to_numpy()).any()),
                                                       other_type_hit=bool((hit & ~known.type.eq(hidden).to_numpy()).any())))
                print("fitted", hidden, fold, "background", len(train_bg), "reference", len(references), "calibration", len(val_bg), flush=True)
            except ValueError as error:
                failures.append(dict(hidden_type=hidden, fold=fold, reason=str(error),
                                     background=len(train_bg), references=len(references), val_bg=len(val_bg)))
                print("unavailable", failures[-1], flush=True)
    frame = pd.DataFrame(records)
    expected_known = int(eligible[:nk].sum())
    evaluated = frame[frame.actual_type != "background"].groupby(["hidden_type", "repeat", "method"]).item_id.nunique()
    complete = bool(len(evaluated) and evaluated.eq(expected_known).all())
    assert complete or failures, "Missing evaluation records must not be reported as successful coverage"
    frame.to_csv(out / "point_scores.csv", index=False)
    pd.DataFrame(recall).to_csv(out / "budget_recall.csv", index=False)
    pd.DataFrame(selections).to_csv(out / "selected_scan_windows.csv", index=False)
    pd.DataFrame(confounds).to_csv(out / "nuisance_correlations.csv", index=False)
    write_json(out / "fits.json", fits)
    write_json(out / "failures.json", failures)
    metrics = []
    for (hidden, repeat, method), part in frame.groupby(["hidden_type", "repeat", "method"]):
        unknown = part.actual_type == hidden
        for negative in ("allowed_known", "background"):
            mask = unknown | (part.actual_type.eq("background") if negative == "background" else (~unknown & ~part.actual_type.eq("background")))
            subset = part[mask]
            labels = subset.actual_type.eq(hidden).to_numpy()
            estimate = grouped_auc(labels, subset.score.to_numpy(), subset.group.to_numpy(), cfg["seed"], cfg["bootstrap_draws"])
            metrics.append(dict(hidden_type=hidden, repeat=repeat, method=method, negative=negative,
                                n_hidden=int(labels.sum()), n_negative=int((~labels).sum()),
                                average_precision=float(average_precision_score(labels, subset.score)),
                                hidden_detection=float(subset.loc[labels, "flagged"].mean()),
                                false_positive_rate=float(subset.loc[~labels, "flagged"].mean()), **estimate))
    pd.DataFrame(metrics).to_csv(out / "novelty_summary.csv", index=False)
    write_json(out / "status.json", dict(fits=len(fits), failed_fits=len(failures), hidden_types=sorted(known.type.unique()),
                                         known_geometrically_eligible=int(known.eligible.sum()), known_eligible=expected_known, known_total=len(known), scan_eligible=int(eligible[nk:].sum()),
                                         unassigned_tail_inputs_used=bool(any((folds[np.asarray(fit[key], dtype=int)] < 0).any()
                                                                            for fit in fits for key in ("representation_training_indices", "calibration_background_indices", "calibration_reference_indices"))),
                                         complete_evaluation_coverage=complete,
                                         full_success_no_missing_folds=not failures and complete,
                                         repeat2_joint_is_independent_validation=False,
                                         interpretation="Empirical ranks, no calibrated genome-wide p-values; no confirmed novel biological class"))


if __name__ == "__main__":
    main()

"""Matched 3.2 kb versus frozen 6.4 kb lattice task-2 evaluation.

WT repeat 1 fits and selects each fold's morphology rule; repeat 2 is an
independent contact-count transfer check. The forest is a known-structure
detector control, not a novelty claim. Test loci were observed previously.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import HuberRegressor

from dlpro.coverage_support import coarse_support
from dlpro.experiment_core import LENGTH, hits, save_json
from dlpro.refinement import morphology_features


OUT = Path("results/discovery_dense_cv")
BUDGETS = (64, 128)


def select(score, centers, budget):
    available = np.ones(len(centers), bool)
    chosen = []
    for i in np.argsort(-score, kind="stable"):
        if not available[i]:
            continue
        chosen.append(int(i))
        distance = abs(centers - centers[i])
        available &= np.minimum(distance, LENGTH - distance) >= 12800
        if len(chosen) == budget:
            return np.asarray(chosen)
    raise ValueError("Insufficient separated scan centers")


def score_features(raw, rows, expected):
    mask = rows[:, :, :, None] & rows[:, :, None, :]
    features = morphology_features(raw, expected, rows)
    density = np.log(
        np.maximum(
            (raw[:, 0] * mask[:, 0]).sum((1, 2))
            / np.maximum(mask[:, 0].sum((1, 2)), 1), 1e-12,
        )
    )
    return features, density


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    known = pd.read_csv("data/processed/known_windows.csv")
    scan = pd.read_csv("data/processed/scan_dense_windows.csv")
    centers = scan.center.to_numpy()
    assert len(scan) == 1451 and np.array_equal(centers[::2], pd.read_csv(
        "data/processed/scan_windows.csv"
    ).center.to_numpy())
    folds = (
        pd.read_csv("results/classification_cv/out_of_fold_predictions.csv")
        .set_index("structure_id").loc[known.structure_id, "fold"].to_numpy()
    )
    distance = abs(centers[:, None] - known.center.to_numpy()[None])
    distance = np.minimum(distance, LENGTH - distance)
    hit = distance <= 6400
    raw = {
        repeat: np.load(f"data/processed/scan_dense_rep{repeat}.npz")["raw"]
        for repeat in (1, 2)
    }
    support = {
        repeat: coarse_support(
            np.load(f"data/processed/coverage_rep{repeat}.npz")["coverage"], centers
        ) >= 0.8
        for repeat in (1, 2)
    }
    random_hits = {}
    rng = np.random.default_rng(3200)
    for budget in BUDGETS:
        random_hits[budget] = np.stack(
            [hit[select(rng.random(len(scan)), centers, budget)].any(axis=0)
             for _ in range(500)]
        )
    rows, candidates, choices, overlaps = [], [], [], []
    pixel_distance = abs(np.arange(128)[:, None] - np.arange(128)[None, :])
    for fold in range(5):
        test = folds == fold
        val = folds == (fold + 1) % 5
        train = ~(test | val)
        held = known[test | val]
        training_known = known[train]
        held_overlap = np.asarray(
            [hits(int(c), 25600, held).any() for c in centers]
        )
        training_overlap = np.asarray(
            [hits(int(c), 25600, training_known, buffer=200).any() for c in centers]
        )
        interior = (centers % 100000 >= 12800) & (centers % 100000 < 87200)
        background = ~held_overlap & ~training_overlap & interior
        blocks = np.unique(scan.background_block.to_numpy()[background])
        order = np.random.default_rng(42 + fold).permutation(blocks)
        training_bg = background & scan.background_block.isin(
            order[: max(1, int(0.7 * len(order)))]
        ).to_numpy()
        assert training_bg.sum() >= 20 and not held_overlap[training_bg].any()
        mask1 = support[1][:, :, :, None] & support[1][:, :, None, :]
        expected = np.zeros((2, 128))
        for channel in range(2):
            for delta in range(128):
                values = raw[1][training_bg, channel][:, pixel_distance == delta]
                valid = mask1[training_bg, channel][:, pixel_distance == delta]
                expected[channel, delta] = max(
                    float(values[valid].mean()) if valid.any() else 0, 1e-8
                )
        features = {}
        density = {}
        for repeat in (1, 2):
            features[repeat], density[repeat] = score_features(
                raw[repeat], support[repeat], expected
            )
        standardized = {repeat: np.empty_like(features[repeat]) for repeat in (1, 2)}
        calibrations = []
        for column in range(8):
            response1 = np.log(np.maximum(features[1][:, column], 1e-6))
            fit = HuberRegressor().fit(
                density[1][training_bg, None], response1[training_bg]
            )
            residual = response1[training_bg] - fit.predict(
                density[1][training_bg, None]
            )
            location = float(np.median(residual))
            spread = max(float(1.4826 * np.median(abs(residual - location))), 0.05)
            for repeat in (1, 2):
                response = np.log(np.maximum(features[repeat][:, column], 1e-6))
                standardized[repeat][:, column] = (
                    response - fit.predict(density[repeat][:, None]) - location
                ) / spread
            calibrations.append(
                dict(feature=column, slope=float(fit.coef_[0]),
                     intercept=float(fit.intercept_), location=location, spread=spread)
            )
        assert all(np.isfinite(z).all() for z in standardized.values())
        def aggregations(z):
            return {"maximum": z.max(axis=1), "mean": z.mean(axis=1),
                    "positive_mean": np.maximum(z, 0).mean(axis=1)}
        validation = {
            name: float(hit[select(score, centers, 64)].any(axis=0)[val].mean())
            for name, score in aggregations(standardized[1]).items()
        }
        chosen_aggregation = max(validation, key=validation.get)
        positive = ~held_overlap & hit[:, train].any(axis=1)
        negative = ~held_overlap & ~training_overlap & (
            distance[:, train] >= 12800
        ).all(axis=1)
        fit_rows = positive | negative
        assert positive.sum() >= 20 and negative.sum() >= 20
        forest = RandomForestClassifier(
            n_estimators=200, max_depth=4, min_samples_leaf=8,
            class_weight="balanced_subsample", n_jobs=1, random_state=42 + fold,
        ).fit(standardized[1][fit_rows], positive[fit_rows].astype(int))
        scores = {
            repeat: {
                "morphology": aggregations(standardized[repeat])[chosen_aggregation],
                "forest_known_control": forest.predict_proba(standardized[repeat])[:, 1],
            }
            for repeat in (1, 2)
        }
        np.savez_compressed(
            OUT / f"fold{fold}_scores.npz", expected=expected,
            z_rep1=standardized[1], z_rep2=standardized[2],
            morphology_rep1=scores[1]["morphology"],
            morphology_rep2=scores[2]["morphology"],
            forest_rep1=scores[1]["forest_known_control"],
            forest_rep2=scores[2]["forest_known_control"],
        )
        choices.append(
            dict(fold=fold, n_background=int(training_bg.sum()),
                 n_positive=int(positive.sum()), n_negative=int(negative.sum()),
                 morphology_choice=chosen_aggregation, validation_choices=validation,
                 calibration=calibrations)
        )
        for budget in BUDGETS:
            selections = {}
            for repeat in (1, 2):
                for recipe, score in scores[repeat].items():
                    ids = select(score, centers, budget)
                    selections[(repeat, recipe)] = set(ids)
                    recalled = hit[ids].any(axis=0)
                    for i in np.flatnonzero(test):
                        rows.append(
                            dict(structure_id=known.structure_id.iloc[i],
                                 type=known.type.iloc[i], group=int(known.group.iloc[i]),
                                 fold=fold, budget=budget, repeat=repeat,
                                 recipe=recipe, recalled=bool(recalled[i]),
                                 random_hit_probability=float(random_hits[budget][:, i].mean()))
                        )
                    for rank, index in enumerate(ids, 1):
                        candidates.append(
                            dict(fold=fold, budget=budget, repeat=repeat, recipe=recipe,
                                 rank=rank, scan_index=int(index), center=int(centers[index]),
                                 score=float(score[index]))
                        )
            for recipe in scores[1]:
                left = selections[(1, recipe)]
                right = selections[(2, recipe)]
                overlaps.append(
                    dict(fold=fold, budget=budget, recipe=recipe,
                         intersection=len(left & right),
                         jaccard=len(left & right) / len(left | right))
                )
        print("dense fold", fold, "background", int(training_bg.sum()),
              "validation", validation, flush=True)
    frame = pd.DataFrame(rows)
    assert frame.groupby(["budget", "repeat", "recipe"]).structure_id.nunique().eq(344).all()
    frame.to_csv(OUT / "out_of_fold_recall.csv", index=False)
    pd.DataFrame(candidates).to_csv(OUT / "candidates.csv", index=False)
    pd.DataFrame(overlaps).to_csv(OUT / "candidate_overlap.csv", index=False)
    frame.groupby(["budget", "repeat", "recipe"]).recalled.agg(["mean", "sum"]).to_csv(
        OUT / "summary.csv"
    )
    save_json(OUT / "protocol.json", dict(
        scan_stride_bp=3200, old_stride_bp=6400, budgets=BUDGETS,
        min_center_spacing_bp=12800, known_center_radius_bp=6400,
        random_trials=500, folds=5, fold_protocols=choices,
        background="Training-annotation-free full25.6kb contexts; held validation/test contexts purged; per-fold100kb blocks",
        forest="Fixed shallow known-structure control trained on repeat1 only",
        repeat2="Independent contact-count transfer; never chooses recipes",
        limitations=["Retrospective previously observed labels and test positions",
                     "Known-center recall is not novel-structure discovery",
                     "Two repeats are not independent biological studies"]
    ))
    print(frame.groupby(["budget", "repeat", "recipe"]).recalled.mean().to_string())


if __name__ == "__main__":
    main()

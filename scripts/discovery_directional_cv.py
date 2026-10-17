"""Fixed directional morphology ablation under the original purged folds."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import HuberRegressor

from dlpro.coverage_support import coarse_support
from dlpro.experiment_core import LENGTH, centered_indices, save_json
from dlpro.refinement import morphology_features
from dlpro.ridge_features import directional_features
from discovery_dense_cv import select


OUT = Path("results/discovery_directional_cv")


def aggregations(z):
    return {"maximum": z.max(1), "mean": z.mean(1),
            "positive_mean": np.maximum(z, 0).mean(1)}


def calibrate(features, density, train):
    fitted, standardized = [], {}
    for repeat in (1, 2):
        standardized[repeat] = np.empty_like(features[repeat])
    for column in range(features[1].shape[1]):
        y1 = np.log(np.maximum(features[1][:, column], 1e-6))
        model = HuberRegressor(max_iter=1000).fit(density[1][train, None], y1[train])
        residual = y1[train] - model.predict(density[1][train, None])
        location = float(np.median(residual))
        spread = max(float(1.4826 * np.median(abs(residual - location))), .05)
        for repeat in (1, 2):
            y = np.log(np.maximum(features[repeat][:, column], 1e-6))
            standardized[repeat][:, column] = (y - model.predict(density[repeat][:, None]) - location) / spread
        fitted.append(dict(slope=float(model.coef_[0]), intercept=float(model.intercept_),
                           location=location, spread=spread))
    return standardized, fitted


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    known = pd.read_csv("data/processed/known_windows.csv")
    centers = pd.read_csv("data/processed/scan_windows.csv").center.to_numpy()
    folds = pd.read_csv("results/classification_cv/out_of_fold_predictions.csv").set_index(
        "structure_id"
    ).loc[known.structure_id, "fold"].to_numpy()
    old_protocol = json.loads(Path("results/discovery_support_cv/protocol.json").read_text())
    raw = {r: np.load(f"data/processed/scan_rep{r}.npz")["raw"] for r in (1, 2)}
    support = {r: coarse_support(np.load(f"data/processed/coverage_rep{r}.npz")["coverage"], centers) >= .8
               for r in (1, 2)}
    density = {}
    for repeat in (1, 2):
        mask = support[repeat][:, 0, :, None] & support[repeat][:, 0, None, :]
        density[repeat] = np.log(np.maximum((raw[repeat][:, 0] * mask).sum((1, 2))
                                          / np.maximum(mask.sum((1, 2)), 1), 1e-12))
    distance = abs(centers[:, None] - known.center.to_numpy()[None])
    distance = np.minimum(distance, LENGTH - distance)
    hit = distance <= 6400
    rng = np.random.default_rng(4016)
    random = {budget: np.mean([hit[select(rng.random(len(centers)), centers, budget)].any(0)
                               for _ in range(500)], axis=0) for budget in (64, 128)}
    starts = np.array([centered_indices(int(c))[1] * 200 for c in centers])
    ends = starts + 25600
    rows, candidates, choices = [], [], []
    for fold in range(5):
        parameters = np.load(f"results/discovery_support_cv/fold{fold}_parameters.npz")
        expected = parameters["expected"]
        train_bg = parameters["background_train"]
        new = {}
        cache = OUT / f"fold{fold}_directional_features.npz"
        if cache.exists():
            with np.load(cache) as saved:
                new = {repeat: saved[f"rep{repeat}"] for repeat in (1, 2)}
        else:
            for repeat in (1, 2):
                new[repeat] = directional_features(raw[repeat], expected, support[repeat])
                print("directional features", fold, repeat, flush=True)
            np.savez_compressed(cache, rep1=new[1], rep2=new[2])
        znew, fitted = calibrate(new, density, train_bg)
        zold = {1: parameters["standardized"], 2: np.empty((len(centers), 8))}
        features2 = morphology_features(raw[2], expected, support[2])
        for column, p in enumerate(old_protocol["fold_protocols"][fold]["calibration"]):
            zold[2][:, column] = (np.log(np.maximum(features2[:, column], 1e-6))
                                     - p["slope"] * density[2] - p["intercept"] - p["location"]) / p["spread"]
        combined = {repeat: np.concatenate([zold[repeat], znew[repeat]], 1) for repeat in (1, 2)}
        assert all(np.isfinite(z).all() for z in combined.values())
        val = folds == (fold + 1) % 5
        test = folds == fold
        training = ~(val | test)
        validation = {}
        chosen = {}
        for name, vectors in (("directional", znew), ("combined", combined)):
            validation[name] = {key: float(hit[select(score, centers, 64)].any(0)[val].mean())
                                for key, score in aggregations(vectors[1]).items()}
            chosen[name] = max(validation[name], key=validation[name].get)
        held = known[val | test]
        overlap = ((starts[:, None] < held.window_end.to_numpy()[None])
                   & (ends[:, None] > held.window_start.to_numpy()[None])).any(1)
        eligible = ~overlap & (starts >= 0) & (ends <= LENGTH)
        positive = eligible & hit[:, training].any(1)
        negative = eligible & (distance[:, training] >= 12800).all(1)
        fit = positive | negative
        forest = RandomForestClassifier(n_estimators=200, max_depth=4, min_samples_leaf=8,
                                         class_weight="balanced_subsample", random_state=42 + fold,
                                         n_jobs=1).fit(combined[1][fit], positive[fit].astype(int))
        scores = {}
        for repeat in (1, 2):
            scores[repeat] = {
                "directional_positive_mean": aggregations(znew[repeat])["positive_mean"],
                "directional_validation_selected": aggregations(znew[repeat])[chosen["directional"]],
                "combined_validation_selected": aggregations(combined[repeat])[chosen["combined"]],
                "forest_known_control": forest.predict_proba(combined[repeat])[:, 1],
                "old_morphology": aggregations(zold[repeat])[old_protocol["fold_protocols"][fold]["choice"]],
            }
        np.savez_compressed(OUT / f"fold{fold}_scores.npz", expected=expected,
                            z_rep1=combined[1], z_rep2=combined[2],
                            **{f"{recipe}_rep{repeat}": score for repeat, recipes in scores.items()
                               for recipe, score in recipes.items()})
        choices.append(dict(fold=fold, selected=chosen, validation=validation, calibration=fitted,
                            n_background=int(train_bg.sum()), n_positive=int(positive.sum()),
                            n_negative=int(negative.sum())))
        for repeat, recipes in scores.items():
            for budget in (64, 128):
                for recipe, score in recipes.items():
                    ids = select(score, centers, budget)
                    recalled = hit[ids].any(0)
                    for i in np.flatnonzero(test):
                        rows.append(dict(structure_id=known.structure_id.iloc[i], type=known.type.iloc[i],
                                         group=int(known.group.iloc[i]), fold=fold, budget=budget,
                                         repeat=repeat, recipe=recipe, recalled=bool(recalled[i]),
                                         random_hit_probability=float(random[budget][i])))
                    candidates.extend(dict(fold=fold, budget=budget, repeat=repeat, recipe=recipe,
                                           rank=rank + 1, center=int(centers[index]), scan_index=int(index),
                                           score=float(score[index])) for rank, index in enumerate(ids))
        print("directional fold", fold, chosen, validation, flush=True)
    frame = pd.DataFrame(rows)
    assert frame.groupby(["budget", "repeat", "recipe"]).structure_id.nunique().eq(344).all()
    frame.to_csv(OUT / "out_of_fold_recall.csv", index=False)
    pd.DataFrame(candidates).to_csv(OUT / "candidates.csv", index=False)
    frame.groupby(["budget", "repeat", "recipe"]).recalled.agg(["mean", "sum"]).to_csv(OUT / "summary.csv")
    save_json(OUT / "protocol.json", dict(fold_choices=choices, stride_bp=6400, budgets=[64, 128],
                                         nms_bp=12800, source="Frozen original per-fold purged background fits",
                                         features="40 directional curvature summaries +8 original contrasts",
                                         fixed_primary="directional_positive_mean", repeat2_fitted=False,
                                         retrospective=True, forest_is_known_structure_control=True))
    print(frame.groupby(["budget", "repeat", "recipe"]).recalled.mean().to_string())


if __name__ == "__main__":
    main()

"""Transfer the fixed task-2 ranker from WT repeat 1 to repeat 2.

The forest recipe was explored on repeat 1 in discovery_supervised_ranker.py.
This checks whether its advantage survives independent contact counts at the
same loci. The annotation positions themselves remain previously observed.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from dlpro.coverage_support import coarse_support
from dlpro.experiment_core import LENGTH, centered_indices, save_json
from dlpro.refinement import morphology_features
from discovery_supervised_ranker import select


OUT = Path("results/discovery_repeat_transfer")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    known = pd.read_csv("data/processed/known_windows.csv")
    scan = pd.read_csv("data/processed/scan_windows.csv")
    folds = (
        pd.read_csv("results/classification_cv/out_of_fold_predictions.csv")
        .set_index("structure_id")
        .loc[known.structure_id, "fold"]
        .to_numpy()
    )
    centers = scan.center.to_numpy()
    distance = abs(centers[:, None] - known.center.to_numpy()[None])
    distance = np.minimum(distance, LENGTH - distance)
    hit = distance <= 6400
    starts = np.asarray([centered_indices(int(c))[1] * 200 for c in centers])
    ends = starts + 25600
    raw = np.load("data/processed/scan_rep2.npz")["raw"]
    coverage = np.load("data/processed/coverage_rep2.npz")["coverage"]
    support = coarse_support(coverage, centers) >= 0.8
    mask = support[:, :, :, None] & support[:, :, None, :]
    protocol = json.loads(Path("results/discovery_support_cv/protocol.json").read_text())
    records, overlaps = [], []
    for fold in range(5):
        original = np.load(f"results/discovery_support_cv/fold{fold}_parameters.npz")
        z1 = original["standardized"]
        expected = original["expected"]
        features = morphology_features(raw, expected, support)
        density = np.log(
            np.maximum(
                (raw[:, 0] * mask[:, 0]).sum((1, 2))
                / np.maximum(mask[:, 0].sum((1, 2)), 1),
                1e-12,
            )
        )
        z2 = np.empty_like(features)
        for column, parameters in enumerate(protocol["fold_protocols"][fold]["calibration"]):
            response = np.log(np.maximum(features[:, column], 1e-6))
            z2[:, column] = (
                response - parameters["slope"] * density - parameters["intercept"]
                - parameters["location"]
            ) / parameters["spread"]
        assert np.isfinite(z2).all()

        test = folds == fold
        val = folds == (fold + 1) % 5
        train = ~(test | val)
        held = known[test | val]
        intersects = (
            (starts[:, None] < held.window_end.to_numpy()[None])
            & (ends[:, None] > held.window_start.to_numpy()[None])
        ).any(axis=1)
        eligible = ~intersects & (starts >= 0) & (ends <= LENGTH)
        positive = eligible & hit[:, train].any(axis=1)
        negative = eligible & (distance[:, train] >= 12800).all(axis=1)
        fitting = positive | negative
        forest = RandomForestClassifier(
            n_estimators=200, max_depth=4, min_samples_leaf=8,
            class_weight="balanced_subsample", n_jobs=1, random_state=42 + fold,
        ).fit(z1[fitting], positive[fitting].astype(int))
        choice = protocol["fold_protocols"][fold]["choice"]
        def old_score(z):
            return {"maximum": z.max(axis=1), "mean": z.mean(axis=1),
                    "positive_mean": np.maximum(z, 0).mean(axis=1)}[choice]
        scores = {
            "morphology_rep1": old_score(z1),
            "morphology_rep2": old_score(z2),
            "forest_rep1": forest.predict_proba(z1)[:, 1],
            "forest_rep2": forest.predict_proba(z2)[:, 1],
        }
        for budget in (64, 128):
            selections = {name: select(score, centers, budget) for name, score in scores.items()}
            for recipe, ids in selections.items():
                recalled = hit[ids].any(axis=0)
                for i in np.flatnonzero(test):
                    records.append(
                        dict(structure_id=known.structure_id.iloc[i], type=known.type.iloc[i],
                             group=int(known.group.iloc[i]), fold=fold, budget=budget,
                             recipe=recipe, recalled=bool(recalled[i]))
                    )
            for recipe in ("morphology", "forest"):
                left = set(selections[f"{recipe}_rep1"])
                right = set(selections[f"{recipe}_rep2"])
                overlaps.append(
                    dict(fold=fold, budget=budget, recipe=recipe,
                         intersection=len(left & right), jaccard=len(left & right) / len(left | right))
                )
        print("repeat transfer fold", fold, flush=True)
    frame = pd.DataFrame(records)
    assert frame.groupby(["budget", "recipe"]).structure_id.nunique().eq(344).all()
    frame.to_csv(OUT / "out_of_fold_recall.csv", index=False)
    pd.DataFrame(overlaps).to_csv(OUT / "candidate_overlap.csv", index=False)
    frame.groupby(["budget", "recipe"]).recalled.agg(["mean", "sum"]).to_csv(OUT / "summary.csv")
    intervals = []
    rng = np.random.default_rng(821)
    for budget in (64, 128):
        panel = frame[frame.budget == budget].pivot(
            index=["structure_id", "group"], columns="recipe", values="recalled"
        )
        groups = panel.index.get_level_values("group").to_numpy()
        distinct = np.unique(groups)
        delta = panel.forest_rep2.to_numpy(dtype=float) - panel.morphology_rep2.to_numpy(dtype=float)
        draws = []
        for _ in range(2000):
            sampled = rng.choice(distinct, len(distinct), replace=True)
            ids = np.concatenate([np.flatnonzero(groups == g) for g in sampled])
            draws.append(float(delta[ids].mean()))
        intervals.append(
            dict(budget=budget, delta_center_recall=float(delta.mean()),
                 group95=np.quantile(draws, [0.025, 0.975]).tolist(),
                 n_groups=len(distinct))
        )
    save_json(OUT / "uncertainty.json", intervals)
    save_json(
        OUT / "protocol.json",
        dict(
            training="Fixed forest on repeat1 fold-purged scan features and training annotations",
            transfer="Repeat2 contact windows; repeat1 expectation and density calibration frozen",
            test="Known-center recall at identical 64/128 budgets and 12800bp center separation",
            limitations=["The forest recipe was explored on the same annotated loci in repeat1",
                         "Repeat2 provides independent contact counts, not independent biological labels",
                         "No novel cluster conclusion follows from known-center recall"],
        ),
    )
    print(frame.groupby(["budget", "recipe"]).recalled.mean().to_string())


if __name__ == "__main__":
    main()

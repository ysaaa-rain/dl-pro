"""Retrospective, fold-purged known-structure ranking control for task 2.

This is a known-structure detector, not evidence of a novel structure class.
The recipes and validation-only selection are fixed in this file before test
fold results are calculated. Existing fold features inherit the original
background-fit protocol and previously observed test positions.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from dlpro.experiment_core import LENGTH, centered_indices, save_json


OUT = Path("results/discovery_supervised_ranker")
BUDGETS = (64, 128)
SEPARATION_BP = 12800
HIT_RADIUS_BP = 6400
NEGATIVE_MARGIN_BP = 12800


def select(score, centers, budget):
    available = np.ones(len(centers), dtype=bool)
    chosen = []
    for i in np.argsort(-score, kind="stable"):
        if not available[i]:
            continue
        chosen.append(int(i))
        distance = abs(centers - centers[i])
        available &= np.minimum(distance, LENGTH - distance) >= SEPARATION_BP
        if len(chosen) == budget:
            return np.asarray(chosen)
    raise ValueError("Insufficient separated scan windows")


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
    hit = distance <= HIT_RADIUS_BP
    starts = np.asarray([centered_indices(int(c))[1] * 200 for c in centers])
    ends = starts + 25600
    old_protocol = json.loads(Path("results/discovery_support_cv/protocol.json").read_text())
    predictions, candidates, decisions = [], [], []

    for fold in range(5):
        test = folds == fold
        val = folds == (fold + 1) % 5
        train = ~(test | val)
        held = known[test | val]
        intersects_held = (
            (starts[:, None] < held.window_end.to_numpy()[None])
            & (ends[:, None] > held.window_start.to_numpy()[None])
        ).any(axis=1)
        eligible = ~intersects_held & (starts >= 0) & (ends <= LENGTH)
        positive = eligible & hit[:, train].any(axis=1)
        negative = eligible & (distance[:, train] >= NEGATIVE_MARGIN_BP).all(axis=1)
        assert positive.sum() >= 20 and negative.sum() >= 20
        assert not intersects_held[positive | negative].any()
        x = np.load(f"results/discovery_support_cv/fold{fold}_parameters.npz")[
            "standardized"
        ]
        assert x.shape == (len(scan), 8) and np.isfinite(x).all()
        choice = old_protocol["fold_protocols"][fold]["choice"]
        morphology = {
            "maximum": x.max(axis=1),
            "mean": x.mean(axis=1),
            "positive_mean": np.maximum(x, 0).mean(axis=1),
        }[choice]
        train_rows = positive | negative
        y = positive[train_rows].astype(int)
        recipes = {
            "morphology": morphology,
            "logistic": LogisticRegression(
                C=0.3, class_weight="balanced", max_iter=500, random_state=42
            )
            .fit(x[train_rows], y)
            .predict_proba(x)[:, 1],
            "forest": RandomForestClassifier(
                n_estimators=200,
                max_depth=4,
                min_samples_leaf=8,
                class_weight="balanced_subsample",
                n_jobs=1,
                random_state=42 + fold,
            )
            .fit(x[train_rows], y)
            .predict_proba(x)[:, 1],
        }
        validation = {
            name: float(hit[select(score, centers, 64)].any(axis=0)[val].mean())
            for name, score in recipes.items()
        }
        # Fixed tie order keeps the existing unsupervised score when validation
        # provides no evidence for the additional supervised complexity.
        selected = max(recipes, key=lambda name: validation[name])
        decisions.append(
            dict(
                fold=fold,
                selected=selected,
                n_positive=int(positive.sum()),
                n_negative=int(negative.sum()),
                **{f"validation64_{name}": score for name, score in validation.items()},
            )
        )
        for name, score in recipes.items():
            for budget in BUDGETS:
                ids = select(score, centers, budget)
                recalled = hit[ids].any(axis=0)
                for i in np.flatnonzero(test):
                    predictions.append(
                        dict(
                            structure_id=known.structure_id.iloc[i],
                            type=known.type.iloc[i],
                            group=int(known.group.iloc[i]),
                            fold=fold,
                            budget=budget,
                            recipe=name,
                            recalled=bool(recalled[i]),
                        )
                    )
                if name == selected:
                    for rank, i in enumerate(ids, 1):
                        candidates.append(
                            dict(
                                fold=fold,
                                budget=budget,
                                rank=rank,
                                scan_index=int(i),
                                center=int(centers[i]),
                                score=float(score[i]),
                                selected_recipe=name,
                            )
                        )

    frame = pd.DataFrame(predictions)
    decisions = pd.DataFrame(decisions)
    selected = frame.merge(decisions[["fold", "selected"]], on="fold")
    selected = selected[selected.recipe == selected.selected].copy()
    selected["recipe"] = "validation_selected"
    frame = pd.concat([frame, selected[frame.columns]], ignore_index=True)
    assert frame.groupby(["budget", "recipe"]).structure_id.nunique().eq(344).all()
    frame.to_csv(OUT / "out_of_fold_recall.csv", index=False)
    decisions.to_csv(OUT / "fold_validation.csv", index=False)
    pd.DataFrame(candidates).to_csv(OUT / "candidates.csv", index=False)
    summary = frame.groupby(["budget", "recipe"]).recalled.agg(["mean", "sum"])
    summary.rename(columns={"mean": "center_recall", "sum": "n_recalled"}).to_csv(
        OUT / "summary.csv"
    )
    uncertainty = []
    rng = np.random.default_rng(963)
    for budget in BUDGETS:
        panel = frame[frame.budget == budget].pivot(
            index=["structure_id", "group"], columns="recipe", values="recalled"
        )
        groups = panel.index.get_level_values("group").to_numpy()
        distinct = np.unique(groups)
        for recipe in ("logistic", "forest", "validation_selected"):
            delta = panel[recipe].to_numpy(dtype=float) - panel["morphology"].to_numpy(
                dtype=float
            )
            draws = []
            for _ in range(2000):
                sampled = rng.choice(distinct, len(distinct), replace=True)
                ids = np.concatenate([np.flatnonzero(groups == g) for g in sampled])
                draws.append(float(delta[ids].mean()))
            uncertainty.append(
                dict(
                    budget=budget,
                    recipe=recipe,
                    delta_center_recall=float(delta.mean()),
                    group95=np.quantile(draws, [0.025, 0.975]).tolist(),
                    n_groups=len(distinct),
                )
            )
    save_json(OUT / "uncertainty.json", uncertainty)
    save_json(
        OUT / "protocol.json",
        dict(
            purpose="Known-structure ranking control, not novel-class confirmation",
            input="Existing eight support-aware morphology features per spatial fold",
            labels="Training-fold annotations only, 6400 bp center radius",
            negatives="At least 12800 bp from training-fold centers",
            purge="Exclude scan windows intersecting validation or test maximum input extents",
            recipes=["morphology", "logistic", "forest"],
            selection="Per-fold validation recall at 64 candidates; ties favor morphology",
            budgets=BUDGETS,
            separation_bp=SEPARATION_BP,
            limitations=[
                "Previously observed test positions; retrospective comparison",
                "Existing feature construction uses the original background protocol",
                "Known-like ranking can suppress genuinely novel morphologies",
                "Center recall does not establish exact boundaries or new biology",
            ],
        ),
    )
    print(summary.to_string())


if __name__ == "__main__":
    main()

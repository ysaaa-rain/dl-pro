"""Test a 50 bp morphology view under the frozen task-2 spatial folds.

Only WT repeat 1 scores candidates. Repeat 2 remains available for later
reproducibility checks. This is retrospective because the loci were exposed
in earlier rounds.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor

from dlpro.experiment_core import LENGTH, save_json
from dlpro.refinement import morphology_features


OUT = Path("results/discovery_fine_scale")
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
    raise ValueError("Insufficient separated scan windows")


def fine_support(coverage, centers):
    native_ids = np.arange(len(coverage))
    width = np.minimum(10, LENGTH - native_ids * 10)
    coarse = native_ids // 5
    total = (LENGTH + 49) // 50
    observed = np.bincount(coarse, weights=width * (coverage > 0), minlength=total)
    area = np.bincount(coarse, weights=width, minlength=total)
    fraction = observed / area
    starts = np.floor((centers + 25) / 50).astype(int) - 64
    return fraction[(starts[:, None] + np.arange(128)[None]) % total][:, None]


def standardized_fine(raw, support, training):
    rows = support >= 0.8
    mask = rows[:, :, :, None] & rows[:, :, None, :]
    distance = abs(np.arange(128)[:, None] - np.arange(128)[None, :])
    expected = np.zeros((1, 128))
    for delta in range(128):
        values = raw[training, 0][:, distance == delta]
        valid = mask[training, 0][:, distance == delta]
        expected[0, delta] = max(float(values[valid].mean()) if valid.any() else 0, 1e-8)
    features = morphology_features(raw, expected, rows)
    density = np.log(
        np.maximum(
            (raw[:, 0] * mask[:, 0]).sum((1, 2))
            / np.maximum(mask[:, 0].sum((1, 2)), 1),
            1e-12,
        )
    )
    z = np.empty_like(features)
    for column in range(features.shape[1]):
        response = np.log(np.maximum(features[:, column], 1e-6))
        fit = HuberRegressor().fit(density[training, None], response[training])
        residual = response[training] - fit.predict(density[training, None])
        location = float(np.median(residual))
        spread = max(float(1.4826 * np.median(abs(residual - location))), 0.05)
        z[:, column] = (response - fit.predict(density[:, None]) - location) / spread
    assert np.isfinite(z).all()
    return z, expected


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
    hit = np.minimum(distance, LENGTH - distance) <= 6400
    raw = np.load("data/processed/fine_scan_rep1.npz")["raw"]
    support = fine_support(
        np.load("data/processed/coverage_rep1.npz")["coverage"], centers
    )
    old_protocol = json.loads(Path("results/discovery_support_cv/protocol.json").read_text())
    rows, candidates, decisions = [], [], []
    for fold in range(5):
        existing = np.load(f"results/discovery_support_cv/fold{fold}_parameters.npz")
        training = existing["background_train"]
        zold = existing["standardized"]
        zfine, expected = standardized_fine(raw, support, training)
        np.savez_compressed(OUT / f"fold{fold}_fine.npz", z=zfine, expected=expected)
        old_choice = old_protocol["fold_protocols"][fold]["choice"]
        old = {
            "maximum": zold.max(axis=1),
            "mean": zold.mean(axis=1),
            "positive_mean": np.maximum(zold, 0).mean(axis=1),
        }[old_choice]
        views = np.stack(
            [zfine.max(axis=1), zold[:, :4].max(axis=1), zold[:, 4:].max(axis=1)],
            axis=1,
        )
        recipes = {
            "existing": old,
            "fine_only": views[:, 0],
            "three_view_median": np.median(views, axis=1),
            "fine_old_blend": 0.5 * views[:, 0] + 0.5 * old,
        }
        test = folds == fold
        val = folds == (fold + 1) % 5
        validation = {
            name: float(hit[select(score, centers, 64)].any(axis=0)[val].mean())
            for name, score in recipes.items()
        }
        selected = max(recipes, key=lambda name: validation[name])
        decisions.append(
            dict(fold=fold, selected=selected, n_background_train=int(training.sum()),
                 **{f"validation64_{name}": value for name, value in validation.items()})
        )
        for name, score in recipes.items():
            for budget in BUDGETS:
                ids = select(score, centers, budget)
                recalled = hit[ids].any(axis=0)
                for i in np.flatnonzero(test):
                    rows.append(
                        dict(structure_id=known.structure_id.iloc[i], type=known.type.iloc[i],
                             group=int(known.group.iloc[i]), fold=fold, budget=budget,
                             recipe=name, recalled=bool(recalled[i]))
                    )
                if name == selected:
                    for rank, i in enumerate(ids, 1):
                        candidates.append(
                            dict(fold=fold, budget=budget, rank=rank, scan_index=int(i),
                                 center=int(centers[i]), score=float(score[i]),
                                 selected_recipe=name)
                        )
        print("fine fold", fold, "validation", validation, "selected", selected, flush=True)
    table = pd.DataFrame(rows)
    decisions = pd.DataFrame(decisions)
    selected = table.merge(decisions[["fold", "selected"]], on="fold")
    selected = selected[selected.recipe == selected.selected].copy()
    selected["recipe"] = "validation_selected"
    table = pd.concat([table, selected[table.columns]], ignore_index=True)
    assert table.groupby(["budget", "recipe"]).structure_id.nunique().eq(344).all()
    table.to_csv(OUT / "out_of_fold_recall.csv", index=False)
    decisions.to_csv(OUT / "fold_validation.csv", index=False)
    pd.DataFrame(candidates).to_csv(OUT / "candidates.csv", index=False)
    table.groupby(["budget", "recipe"]).recalled.agg(["mean", "sum"]).to_csv(
        OUT / "summary.csv"
    )
    save_json(
        OUT / "protocol.json",
        dict(
            purpose="50bp shape view with fixed validation-only recipe selection",
            data="WT repeat1 fine scan; repeat2 excluded from score and selection",
            expected="Distance profiles from existing fold-purged training background",
            features="Positive local O/E contrasts at 50/100/200bp; each view density calibrated on training background",
            recipes=list(recipes), budgets=BUDGETS, separation_bp=12800,
            limitations=["Previously observed loci; retrospective comparison",
                         "Existing training background inherits earlier annotation policy",
                         "Center recall does not establish novel structure classes"],
        ),
    )
    print(table.groupby(["budget", "recipe"]).recalled.mean())


if __name__ == "__main__":
    main()

"""Single exploratory positive-type weighting control for known detection."""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from dlpro.experiment_core import LENGTH, centered_indices, save_json
from discovery_dense_cv import select
from summarize_dense_discovery import bootstrap


OUT = Path("results/discovery_type_weight_control")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    known = pd.read_csv("data/processed/known_windows.csv")
    centers = pd.read_csv("data/processed/scan_windows.csv").center.to_numpy()
    folds = pd.read_csv("results/classification_cv/out_of_fold_predictions.csv").set_index(
        "structure_id").loc[known.structure_id, "fold"].to_numpy()
    distance = abs(centers[:, None] - known.center.to_numpy()[None])
    distance = np.minimum(distance, LENGTH - distance)
    hit = distance <= 6400
    starts = np.array([centered_indices(int(c))[1] * 200 for c in centers])
    ends = starts + 25600
    records, choices, candidates = [], [], []
    for fold in range(5):
        test = folds == fold
        val = folds == (fold + 1) % 5
        training = ~(test | val)
        held = known[val | test]
        overlap = ((starts[:, None] < held.window_end.to_numpy()[None])
                   & (ends[:, None] > held.window_start.to_numpy()[None])).any(1)
        eligible = ~overlap & (starts >= 0) & (ends <= LENGTH)
        positive = eligible & hit[:, training].any(1)
        negative = eligible & (distance[:, training] >= 12800).all(1)
        fit = positive | negative
        nearest = distance.astype(float)
        nearest[:, ~training] = np.inf
        target_types = known.type.to_numpy()[nearest.argmin(1)]
        counts = pd.Series(target_types[positive]).value_counts()
        assert len(counts) == 3
        weights = np.ones(len(centers))
        for label, count in counts.items():
            weights[positive & (target_types == label)] = positive.sum() / (3 * count)
        with np.load(f"results/discovery_directional_cv/fold{fold}_scores.npz") as data:
            vectors = {r: data[f"z_rep{r}"] for r in (1, 2)}
            original = {r: data[f"forest_known_control_rep{r}"] for r in (1, 2)}
        model = RandomForestClassifier(n_estimators=200, max_depth=4, min_samples_leaf=8,
                                        class_weight="balanced_subsample", random_state=42 + fold,
                                        n_jobs=1).fit(vectors[1][fit], positive[fit].astype(int),
                                                      sample_weight=weights[fit])
        choices.append(dict(fold=fold, positive_type_counts=counts.to_dict(),
                            positive_weights={k: float(positive.sum() / (3*v)) for k, v in counts.items()},
                            n_positive=int(positive.sum()), n_negative=int(negative.sum())))
        for repeat in (1, 2):
            recipes = {"type_weighted": model.predict_proba(vectors[repeat])[:, 1],
                       "unweighted": original[repeat]}
            for recipe, score in recipes.items():
                for budget in (64, 128):
                    ids = select(score, centers, budget)
                    recalled = hit[ids].any(0)
                    for i in np.flatnonzero(test):
                        records.append(dict(structure_id=known.structure_id.iloc[i], type=known.type.iloc[i],
                                            group=int(known.group.iloc[i]), fold=fold, budget=budget,
                                            repeat=repeat, recipe=recipe, recalled=bool(recalled[i])))
                    candidates.extend(dict(fold=fold, budget=budget, repeat=repeat, recipe=recipe,
                                           rank=rank+1, center=int(centers[index]), score=float(score[index]))
                                      for rank, index in enumerate(ids))
        print("positive type control fold", fold, counts.to_dict(), flush=True)
    frame = pd.DataFrame(records)
    frame.to_csv(OUT / "out_of_fold_recall.csv", index=False)
    pd.DataFrame(candidates).to_csv(OUT / "candidates.csv", index=False)
    frame.groupby(["budget", "repeat", "recipe", "type"]).recalled.mean().to_csv(OUT / "per_type.csv")
    summary = frame.groupby(["budget", "repeat", "recipe"]).recalled.mean().rename("micro_recall").to_frame()
    summary["macro_type_recall"] = frame.groupby(["budget", "repeat", "recipe", "type"]).recalled.mean().groupby(
        ["budget", "repeat", "recipe"]).mean()
    summary.to_csv(OUT / "summary.csv")
    intervals = {}
    for (budget, repeat), group in frame.groupby(["budget", "repeat"]):
        panel = group.pivot(index=["structure_id", "type", "group"], columns="recipe", values="recalled")
        intervals[f"budget{budget}_repeat{repeat}_micro"] = bootstrap(panel, "type_weighted", "unweighted")
    save_json(OUT / "paired_uncertainty.json", intervals)
    save_json(OUT / "protocol.json", dict(
        fixed_after_observed_type_tradeoff=True, n_trees=200, max_depth=4, min_leaf=8,
        feature_source="Frozen directional48-dimensional OOF scan features",
        sole_change="Positive sample weights inverse nearest-training-annotation type count; three type totals equal",
        inference_metadata=False, no_recipe_selection=True, fold_fits=choices,
        limitation="Exploratory known-structure detection; no novel-cluster inference",
    ))
    print(summary.to_string())


if __name__ == "__main__":
    main()

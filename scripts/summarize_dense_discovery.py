"""Paired, spatial-group-aware summary of frozen 3.2 kb scan experiments."""

import numpy as np
import pandas as pd

from dlpro.experiment_core import save_json


def bootstrap(frame, left, right):
    delta = frame[left].to_numpy(dtype=float) - frame[right].to_numpy(dtype=float)
    groups = frame.index.get_level_values("group").to_numpy()
    distinct = np.unique(groups)
    rng = np.random.default_rng(3201)
    draws = []
    for _ in range(2000):
        chosen = rng.choice(distinct, len(distinct), replace=True)
        ids = np.concatenate([np.flatnonzero(groups == group) for group in chosen])
        draws.append(float(delta[ids].mean()))
    return dict(delta=float(delta.mean()), group95=np.quantile(draws, [0.025, 0.975]).tolist(),
                n_groups=len(distinct))


def main():
    old = pd.read_csv("results/discovery_support_cv/out_of_fold_recall.csv")
    new = pd.read_csv("results/discovery_dense_cv/out_of_fold_recall.csv")
    old = old.rename(columns={"recalled": "old_morphology"})
    data = new.pivot(index=["structure_id", "type", "group", "fold", "budget"],
                     columns=["repeat", "recipe"], values="recalled")
    data.columns = [f"dense_repeat{repeat}_{recipe}" for repeat, recipe in data.columns]
    data = data.reset_index().merge(old, on=["structure_id", "type", "group", "fold", "budget"],
                                    validate="one_to_one")
    random = new[new.repeat == 1].drop_duplicates(["structure_id", "budget"])[
        ["structure_id", "budget", "random_hit_probability"]
    ]
    data = data.merge(random, on=["structure_id", "budget"], validate="one_to_one")
    data.to_csv("results/discovery_dense_cv/paired_comparison.csv", index=False)
    summaries = []
    intervals = {}
    for budget in (64, 128):
        subset = data[data.budget == budget]
        assert len(subset) == 344
        for kind, part in [("all", subset), *[(label, group) for label, group in subset.groupby("type")]]:
            summaries.append(dict(budget=budget, type=kind, n=len(part),
                                  old_morphology=float(part.old_morphology.mean()),
                                  dense_morphology_rep1=float(part.dense_repeat1_morphology.mean()),
                                  dense_morphology_rep2=float(part.dense_repeat2_morphology.mean()),
                                  dense_forest_rep1=float(part.dense_repeat1_forest_known_control.mean()),
                                  dense_forest_rep2=float(part.dense_repeat2_forest_known_control.mean()),
                                  random_mean=float(part.random_hit_probability.mean())))
        panel = subset.set_index(["structure_id", "group"])
        for left, right in (
            ("dense_repeat1_morphology", "old_morphology"),
            ("dense_repeat2_morphology", "old_morphology"),
            ("dense_repeat1_forest_known_control", "old_morphology"),
            ("dense_repeat2_forest_known_control", "old_morphology"),
            ("dense_repeat1_morphology", "dense_repeat2_morphology"),
        ):
            intervals[f"budget{budget}_{left}_minus_{right}"] = bootstrap(panel, left, right)
    pd.DataFrame(summaries).to_csv("results/discovery_dense_cv/paired_summary.csv", index=False)
    save_json("results/discovery_dense_cv/paired_uncertainty.json", intervals)
    overlap = pd.read_csv("results/discovery_dense_cv/candidate_overlap.csv")
    overlap.groupby(["budget", "recipe"])[["intersection", "jaccard"]].mean().to_csv(
        "results/discovery_dense_cv/repeat_stability.csv"
    )
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()

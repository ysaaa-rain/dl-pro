"""Paired spatial uncertainty, class recalls and candidate-repeat stability."""

from pathlib import Path
import numpy as np
import pandas as pd

from dlpro.experiment_core import save_json
from summarize_dense_discovery import bootstrap


OUT = Path("results/discovery_directional_cv")


def main():
    frame = pd.read_csv(OUT / "out_of_fold_recall.csv")
    old = pd.read_csv("results/discovery_repeat_transfer/out_of_fold_recall.csv")
    summaries, intervals = [], {}
    for budget in (64, 128):
        for repeat in (1, 2):
            subset = frame[(frame.budget == budget) & (frame.repeat == repeat)]
            panel = subset.pivot(index=["structure_id", "type", "group", "fold"],
                                  columns="recipe", values="recalled")
            old_forest = old[(old.budget == budget) & (old.recipe == f"forest_rep{repeat}")]
            old_forest = old_forest.set_index(["structure_id", "type", "group", "fold"])
            panel["old_forest_control"] = old_forest.recalled
            assert len(panel) == 344 and panel.notna().all().all()
            for recipe in ("directional_positive_mean", "directional_validation_selected",
                           "combined_validation_selected", "forest_known_control"):
                intervals[f"budget{budget}_repeat{repeat}_{recipe}_minus_old_morphology"] = bootstrap(
                    panel, recipe, "old_morphology"
                )
            intervals[f"budget{budget}_repeat{repeat}_newforest_minus_oldforest"] = bootstrap(
                panel, "forest_known_control", "old_forest_control"
            )
            for label in ("all", "CHID", "CHIN", "OPCID"):
                part = panel if label == "all" else panel[panel.index.get_level_values("type") == label]
                summaries.append(dict(budget=budget, repeat=repeat, type=label, n=len(part),
                                      **{key: float(part[key].mean()) for key in panel.columns}))
    pd.DataFrame(summaries).to_csv(OUT / "paired_summary.csv", index=False)
    save_json(OUT / "paired_uncertainty.json", intervals)
    candidates = pd.read_csv(OUT / "candidates.csv")
    stability = []
    for (fold, budget, recipe), group in candidates.groupby(["fold", "budget", "recipe"]):
        one = set(group[group.repeat == 1].center)
        two = set(group[group.repeat == 2].center)
        stability.append(dict(fold=fold, budget=budget, recipe=recipe,
                              intersection=len(one & two), jaccard=len(one & two) / len(one | two)))
    pd.DataFrame(stability).to_csv(OUT / "candidate_stability.csv", index=False)
    pd.DataFrame(stability).groupby(["budget", "recipe"]).jaccard.mean().to_csv(
        OUT / "repeat_stability_summary.csv"
    )
    print(pd.DataFrame(summaries).query('type == "all"').to_string(index=False))


if __name__ == "__main__":
    main()

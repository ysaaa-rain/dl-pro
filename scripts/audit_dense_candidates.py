"""Audit dense morphology consensus without upgrading candidates to new classes.

Nearby centers from different folds form one genomic candidate. Only WT
repeat-supported, full-context-unannotated, independent members may enter the
fixed six-cluster exploratory check. The criteria mirror the prior audit.
"""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from dlpro.coverage_support import coarse_support
from dlpro.experiment_core import LENGTH, hits, save_json


OUT = Path("results/discovery_dense_cv")


def circular_distance(a, b):
    distance = abs(int(a) - int(b))
    return min(distance, LENGTH - distance)


def regions(frame):
    frame = frame.sort_values("center").reset_index(drop=True)
    groups = [[0]]
    for i in range(1, len(frame)):
        if frame.center.iloc[i] - frame.center.iloc[i - 1] <= 6400:
            groups[-1].append(i)
        else:
            groups.append([i])
    if len(groups) > 1 and circular_distance(frame.center.iloc[groups[0][0]],
                                              frame.center.iloc[groups[-1][-1]]) <= 6400:
        groups[0] = groups[-1] + groups[0]
        groups.pop()
    result = []
    for members in groups:
        part = frame.iloc[members]
        if part.fold.nunique() < 3:
            continue
        representative = int(part.sort_values(["score", "center"], ascending=[False, True]).center.iloc[0])
        result.append(dict(center=representative, selected_folds=int(part.fold.nunique()),
                           component_centers=";".join(map(str, sorted(part.center.unique()))),
                           mean_score=float(part.score.mean())))
    return pd.DataFrame(result).sort_values("center").reset_index(drop=True)


def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/discovery_dense_cv")
    parser.add_argument("--scan-prefix", default="scan_dense")
    parser.add_argument("--recipe", default="morphology")
    parser.add_argument("--id-prefix", default="D")
    args = parser.parse_args()
    OUT = Path(args.results)
    selections = pd.read_csv(OUT / "candidates.csv")
    scan = pd.read_csv(f"data/processed/{args.scan_prefix}_windows.csv")
    known = pd.read_csv("data/annotations/structures.csv")
    primary = selections[(selections.budget == 64) & (selections.recipe == args.recipe)]
    rep1 = regions(primary[primary.repeat == 1])
    rep2 = regions(primary[primary.repeat == 2])
    if rep1.empty:
        raise ValueError("No repeat-1 consensus centers")
    rep1["repeat2_consensus_near"] = [
        any(circular_distance(c, other) <= 6400 for other in rep2.center)
        for c in rep1.center
    ]
    rep1["known_center_12_8kb_overlap"] = [
        bool(hits(int(c), 12800, known, buffer=200).any()) for c in rep1.center
    ]
    rep1["known_full_25_6kb_context_overlap"] = [
        bool(hits(int(c), 25600, known, buffer=200).any()) for c in rep1.center
    ]
    index = scan.set_index("center").loc[rep1.center].index
    ids = scan.index[scan.center.isin(index)].to_numpy()
    assert len(ids) == len(rep1) and np.array_equal(scan.center.iloc[ids], rep1.center)
    with np.load(OUT / "fold0_scores.npz") as saved:
        expected = saved["expected"][0]
        z = saved["z_rep1"][ids]
    d = abs(np.arange(128)[:, None] - np.arange(128)[None, :])
    triangle = np.triu_indices(128, 2)
    support = []
    for repeat in (1, 2):
        with np.load(f"data/processed/coverage_rep{repeat}.npz") as saved:
            support.append(coarse_support(saved["coverage"], rep1.center.to_numpy())[:, 0] >= .8)
    images = []
    for repeat in (1, 2):
        with np.load(f"data/processed/{args.scan_prefix}_rep{repeat}.npz") as saved:
            images.append(saved["raw"][ids, 0])
    correlations, fractions = [], []
    for i in range(len(rep1)):
        common = support[0][i] & support[1][i]
        mask = (common[:, None] & common[None, :])[triangle]
        fractions.append(float(mask.mean()))
        x = (images[0][i] / expected[d])[triangle][mask]
        y = (images[1][i] / expected[d])[triangle][mask]
        correlations.append(float(np.corrcoef(x, y)[0, 1])
                            if len(x) > 2 and x.std() > 0 and y.std() > 0 else np.nan)
    rep1["supported_triangle_fraction"] = fractions
    rep1["wt_oe_correlation_supported"] = correlations
    rep1["repeat_supported"] = (
        (rep1.supported_triangle_fraction >= .5)
        & (rep1.wt_oe_correlation_supported >= .5)
    )
    rep1["eligible_unannotated_context"] = (
        rep1.repeat_supported & ~rep1.known_full_25_6kb_context_overlap
    )
    selected = []
    for row in rep1[rep1.eligible_unannotated_context].itertuples():
        if all(circular_distance(row.center, other) >= 25600 for other in selected):
            selected.append(row.center)
    rep1["independent_context_member"] = rep1.center.isin(selected)
    rep1["candidate_id"] = [f"{args.id_prefix}{i+1:03d}" for i in range(len(rep1))]
    rep1["cluster"] = -1
    if len(rep1) >= 6:
        clustering = KMeans(n_clusters=6, n_init=20, random_state=42).fit(z)
        rep1["cluster"] = clustering.labels_
    cluster_rows = []
    for label, part in rep1.groupby("cluster"):
        members = part[part.independent_context_member]
        cluster_rows.append(dict(cluster=int(label), total_members=len(part),
                                 independent_members=len(members),
                                 repeat2_consensus_members=int(members.repeat2_consensus_near.sum()),
                                 known_context_members=int(part.known_full_25_6kb_context_overlap.sum()),
                                 tentative_new_cluster=bool(label >= 0 and len(members) >= 5
                                                            and members.repeat2_consensus_near.all()
                                                            and not part.known_full_25_6kb_context_overlap.any())))
    rep1.to_csv(OUT / "candidate_context_audit.csv", index=False)
    pd.DataFrame(cluster_rows, columns=["cluster", "total_members", "independent_members",
                                        "repeat2_consensus_members", "known_context_members",
                                        "tentative_new_cluster"]).to_csv(
        OUT / "candidate_clusters.csv", index=False
    )
    save_json(OUT / "candidate_audit_protocol.json", dict(
        selection=f"{args.recipe} 64-budget center selected in >=3 of 5 folds, nearby centers within6400bp merged",
        repeat2_consensus="Same rule applied to independent WT repeat2 count image, center within6400bp",
        support="Both WT repeats >=80% native support rows; >=50% common offdiagonal upper triangle; O/E Pearson >=.5",
        known_exclusion="No annotation overlap with full25.6kb candidate context plus200bp buffer",
        independent_member_rule="Genome-order greedy separation >=25600bp",
        clustering="KMeans six fixed shape-feature groups on all consensus candidates if >=6; tentative cluster needs >=5 independent members, all repeat2 consensus and no known-context members",
        total_consensus=len(rep1), repeat_supported=int(rep1.repeat_supported.sum()),
        full_context_unannotated=int(rep1.eligible_unannotated_context.sum()),
        independent_members=int(rep1.independent_context_member.sum()),
        tentative_new_clusters=int(sum(item["tentative_new_cluster"] for item in cluster_rows)),
        limitations=["Candidate audit is exploratory and uses previously observed loci",
                     "Fold fits share the same biological counts", "Two WT repeats are not external biological confirmation",
                     "NoLig and perturbation controls have not been repeated for newly located candidates"],
    ))
    print(rep1[["candidate_id", "center", "selected_folds", "repeat2_consensus_near",
                "repeat_supported", "known_full_25_6kb_context_overlap",
                "independent_context_member", "cluster"]].to_string(index=False))


if __name__ == "__main__":
    main()

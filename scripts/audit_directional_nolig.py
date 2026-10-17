"""Descriptive NoLig check and historical-region crosswalk for four candidates."""

from pathlib import Path
import numpy as np
import pandas as pd

from dlpro.coverage_support import coarse_support
from dlpro.experiment_core import ContactReader, LENGTH, centered_indices, save_json


OUT = Path("results/discovery_directional_cv")


def correlation(x, y):
    return float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 and x.std() > 0 and y.std() > 0 else None


def main():
    candidates = pd.read_csv(OUT / "candidate_context_audit.csv")
    candidates = candidates[candidates.independent_context_member].copy()
    assert len(candidates) == 4
    samples = pd.read_csv("data/annotations/samples.csv")
    row = samples[samples.condition == "NoLig"].iloc[0]
    reader = ContactReader(row.path)
    scan = pd.read_csv("data/processed/scan_windows.csv")
    ids = [int(scan.index[scan.center == center][0]) for center in candidates.center]
    with np.load(OUT / "fold0_scores.npz") as source:
        expected = source["expected"][0]
    distance = abs(np.arange(128)[:, None] - np.arange(128)[None, :])
    triangle = np.triu_indices(128, 2)
    wt, support = {}, {}
    for repeat in (1, 2):
        with np.load(f"data/processed/scan_rep{repeat}.npz") as source:
            wt[repeat] = source["raw"][ids, 0]
        with np.load(f"data/processed/coverage_rep{repeat}.npz") as source:
            support[repeat] = coarse_support(source["coverage"], candidates.center.to_numpy())[:, 0] >= .8
    historical = pd.read_csv("results/candidate_conditions/unannotated_consensus.csv")
    known = pd.read_csv("data/annotations/structures.csv")
    records = []
    for i, candidate in enumerate(candidates.itertuples()):
        center = int(candidate.center)
        common = support[1][i] & support[2][i]
        mask = (common[:, None] & common[None, :])[triangle]
        negative = reader.window(centered_indices(center)[0], 128, 100)
        a = (wt[1][i] / expected[distance])[triangle][mask]
        b = (wt[2][i] / expected[distance])[triangle][mask]
        c = (negative / expected[distance])[triangle][mask]
        previous_distance = abs(historical.center.to_numpy() - center)
        previous_distance = np.minimum(previous_distance, LENGTH - previous_distance)
        previous = int(previous_distance.argmin())
        gap = np.full(len(known), np.inf)
        for shift in (-LENGTH, 0, LENGTH):
            left = known.start.to_numpy() + shift
            right = known.end.to_numpy() + shift
            gap = np.minimum(gap, np.maximum(np.maximum(left - (center + 12800),
                                                        (center - 12800) - right), 0))
        near = int(gap.argmin())
        records.append(dict(candidate_id=candidate.candidate_id, center=center,
                            wt_repeat_oe_correlation=correlation(a, b),
                            nolig_wt_oe_correlation=correlation(c, (a+b)/2),
                            nolig_wt_logoe_correlation=correlation(np.log1p(c), np.log1p((a+b)/2)),
                            supported_triangle_fraction=float(mask.mean()),
                            nolig_density=float(negative[triangle][mask].mean()),
                            wt_density=float(((wt[1][i]+wt[2][i])/2)[triangle][mask].mean()),
                            nearest_old_candidate=historical.candidate_id.iloc[previous],
                            old_center=int(historical.center.iloc[previous]),
                            old_center_distance_bp=int(previous_distance[previous]),
                            old_region_within6400bp=bool(previous_distance[previous] <= 6400),
                            nearest_annotation=known.structure_id.iloc[near],
                            full_context_edge_gap_bp=int(gap[near])))
    result = pd.DataFrame(records)
    result.to_csv(OUT / "candidate_nolig_and_history.csv", index=False)
    save_json(OUT / "candidate_nolig_protocol.json", dict(
        negative_sample=row.sample_id, source_sha256=row.sha256,
        negative_repeats=1, source_library_pairs=int(row.library_pairs),
        source="Official NoLig Cooler; registered gzip and decompressed SHA256 both verified",
        mask="Identical WTrep1/rep2 common observed support,80%native bin threshold,upper triangle offset>=2",
        normalization="Each library per-million pairs; common fold0 WT training distance expectation",
        history_regions_within6400bp=int(result.old_region_within6400bp.sum()),
        limitations=["NoLig has one repeat and different library composition and depth",
                     "Low NoLig similarity does not establish novel biology or causality",
                     "Shifted centers can change known-context eligibility without introducing a new underlying structure"],
    ))
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()

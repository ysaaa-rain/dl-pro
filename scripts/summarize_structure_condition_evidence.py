"""Combine retained structure-condition hits and audit their evidence sources."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import cooler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dlpro.data import overlap_groups


ROOT = Path("results/structure_conditions")
OUT = ROOT / "evidence_synthesis"
STRUCTURES = Path("data/annotations/structures.csv")
SAMPLES = Path("data/annotations/samples.csv")
MUTANT_RESULTS = ROOT / "robustness/robust_candidates.csv"
HEAT_RESULTS = ROOT / "44C/robustness/robust_candidates.csv"
MIN_TARGET_CHANGE = 0.25
BIN_BP = 100
GENOME_BP = 4_641_652
GENOME_BINS = (GENOME_BP + BIN_BP - 1) // BIN_BP


# Values are taken from the public GEO sample records and linked SRA RunInfo.
# The series-level Cooler supplements do not contain GSM identifiers.
GEO_SAMPLE_ROWS = [
    ("GSM8394770", "37C", "t1", 1, "WT_MicroC_t1_biol_rep1", "GPL32081", "NextSeq 2000", "ag03", "SRX25308045", "SRR29808782", 49_501_958),
    ("GSM8394771", "44C", "t1", 1, "WT_HS_MicroC_t1_biol_rep1", "GPL32081", "NextSeq 2000", "ag04", "SRX25308046", "SRR29808781", 71_482_159),
    ("GSM8394772", "37C", "t1", 2, "WT_MicroC_t1_biol_rep2", "GPL32081", "NextSeq 2000", "ag05", "SRX25308047", "SRR29808780", 54_489_154),
    ("GSM8394773", "44C", "t1", 2, "WT_HS_MicroC_t1_biol_rep2", "GPL32081", "NextSeq 2000", "ag06", "SRX25308048", "SRR29808779", 38_708_745),
    ("GSM8394774", "37C", "t2", 1, "WT_MicroC_t2_biol_rep1", "GPL34700", "DNBSEQ-T7", "ag07", "SRX25308049", "SRR29808778", 609_022_183),
    ("GSM8394775", "44C", "t2", 1, "WT_HS_MicroC_t2_biol_rep1", "GPL34700", "DNBSEQ-T7", "ag08", "SRX25308050", "SRR29808777", 627_206_315),
    ("GSM8394776", "37C", "t2", 2, "WT_MicroC_t2_biol_rep2", "GPL34700", "DNBSEQ-T7", "ag09", "SRX25308051", "SRR29808776", 595_064_245),
    ("GSM8394777", "44C", "t2", 2, "WT_HS_MicroC_t2_biol_rep2", "GPL34700", "DNBSEQ-T7", "ag10", "SRX25308052", "SRR29808775", 637_595_590),
]


def load_candidate_tables() -> pd.DataFrame:
    mutant = pd.read_csv(MUTANT_RESULTS)
    heat = pd.read_csv(HEAT_RESULTS)
    mutant = mutant.loc[mutant.robust_candidate].copy()
    heat = heat.loc[heat.robust_candidate].copy()
    for frame in (mutant, heat):
        if frame.structure_id.duplicated().any() and frame.condition.nunique() == 1:
            raise ValueError("Repeated annotation within one condition")
    if len(mutant) != 10 or len(heat) != 16:
        raise ValueError(f"Expected 10 mutant and 16 heat-shock hits, found {len(mutant)} + {len(heat)}")
    return pd.concat([mutant, heat], ignore_index=True, sort=False)


def density_components(matrix: np.ndarray, feature: pd.Series, all_annotated: np.ndarray) -> tuple[float, float]:
    """Return target and clean-flank CPM densities with the 1x clean protocol."""
    start, end = int(feature.start), int(feature.end)
    length = end - start
    lo = ((start - 2 * length) // BIN_BP) * BIN_BP
    centers = lo + (np.arange(len(matrix)) + 0.5) * BIN_BP
    core = (centers >= start) & (centers < end)
    flank = ((centers >= start - length) & (centers < start)) | ((centers >= end) & (centers < end + length))
    bins = np.clip((centers // BIN_BP).astype(int), 0, len(all_annotated) - 1)
    clean_flank = flank & ~all_annotated[bins]
    max_distance = max(2, int(np.floor(min(length, 0.5 * length) / BIN_BP)))
    target_total = 0.0
    background_scaled = 0.0
    target_pixels = 0
    for distance in range(2, max_distance + 1):
        if distance >= len(matrix):
            break
        values = np.diag(matrix, distance)
        target_pairs = core[:-distance] & core[distance:]
        background_pairs = clean_flank[:-distance] & clean_flank[distance:]
        if target_pairs.any() and background_pairs.any():
            n_target = int(target_pairs.sum())
            background_mean = float(values[background_pairs].mean())
            if background_mean > 0:
                target_total += float(values[target_pairs].sum())
                background_scaled += background_mean * n_target
                target_pixels += n_target
    if not target_pixels or target_total <= 0 or background_scaled <= 0:
        return float("nan"), float("nan")
    return target_total / target_pixels, background_scaled / target_pixels


def summarize_components(
    structure_id: str,
    condition: str,
    features: pd.DataFrame,
    sample_table: pd.DataFrame,
    cache_handles: dict[str, object],
    all_annotated: np.ndarray,
) -> dict[str, object]:
    feature = features.loc[features.structure_id == structure_id].iloc[0]
    wt_samples = sample_table[(sample_table.condition == "37C")].sort_values("replicate")
    condition_samples = sample_table[sample_table.condition == condition].sort_values("replicate")
    if len(wt_samples) != 2 or len(condition_samples) != 2:
        raise ValueError(f"Expected two source matrices for WT and {condition}")

    values: dict[str, dict[str, float]] = {"target": {}, "background": {}}
    sample_ids: dict[str, list[str]] = {"WT": [], condition: []}
    for label, samples in (("WT", wt_samples), (condition, condition_samples)):
        for sample in samples.itertuples():
            if sample.sample_id not in cache_handles:
                raise FileNotFoundError(f"Missing expanded cache for {sample.sample_id}")
            target, background = density_components(
                cache_handles[sample.sample_id][structure_id], feature, all_annotated
            )
            if not (np.isfinite(target) and np.isfinite(background) and target > 0 and background > 0):
                raise ValueError(f"Invalid target/background density for {structure_id}, {sample.sample_id}")
            values["target"][sample.sample_id] = target
            values["background"][sample.sample_id] = background
            sample_ids[label].append(sample.sample_id)

    result: dict[str, object] = {}
    for component in ("target", "background"):
        wt_log = np.log2([values[component][sid] for sid in sample_ids["WT"]])
        condition_log = np.log2([values[component][sid] for sid in sample_ids[condition]])
        cross = (condition_log[:, None] - wt_log[None, :]).ravel()
        result[f"{component}_log2_change"] = float(condition_log.mean() - wt_log.mean())
        result[f"{component}_all_four_same_direction"] = bool(np.all(cross > 0) or np.all(cross < 0))
        result[f"{component}_minimum_cross_effect"] = float(np.min(np.abs(cross)))
        result[f"WT_rep1_{component}_density"] = float(values[component][sample_ids["WT"][0]])
        result[f"WT_rep2_{component}_density"] = float(values[component][sample_ids["WT"][1]])
        result[f"condition_rep1_{component}_density"] = float(values[component][sample_ids[condition][0]])
        result[f"condition_rep2_{component}_density"] = float(values[component][sample_ids[condition][1]])
    result["WT_sample_rep1"] = sample_ids["WT"][0]
    result["WT_sample_rep2"] = sample_ids["WT"][1]
    result["condition_sample_rep1"] = sample_ids[condition][0]
    result["condition_sample_rep2"] = sample_ids[condition][1]
    return result


def interpretation(local: float, target: float, background: float, repeat_stable: bool) -> str:
    same = np.sign(target) == np.sign(local)
    if not repeat_stable:
        return "target-direction evidence is not consistent across all four cross-replicate pairs"
    if not same:
        return "background-driven: target density shifts opposite to local enrichment"
    if abs(target) < MIN_TARGET_CHANGE:
        return "flank-dominant: target-density shift is below 0.25 log2"
    if abs(target) > abs(background):
        return "target-centered: target-density change exceeds flank-density change"
    return "mixed, with larger flank-density contribution"


def write_geo_mapping_audit(sample_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for record in GEO_SAMPLE_ROWS:
        (
            accession,
            condition,
            timepoint,
            replicate,
            title,
            platform,
            instrument,
            library_name,
            experiment,
            run,
            spots,
        ) = record
        cooler_id = f"GSE272159_{condition}_rep{replicate}"
        sample = sample_table.loc[sample_table.sample_id == cooler_id]
        if len(sample) != 1:
            raise ValueError(f"Expected one series-level Cooler record for {cooler_id}")
        cooler_sum = int(sample.iloc[0].library_pairs)
        ratio = cooler_sum / spots
        if timepoint == "t1":
            assessment = "ruled out as one-to-one source: Cooler pair sum is greater than SRA spot count"
        else:
            assessment = "strongly consistent with the matching t2 replicate; no explicit GSM-to-Cooler link"
        rows.append(
            {
                "sample_accession": accession,
                "sample_title": title,
                "condition": condition,
                "timepoint_group": timepoint,
                "replicate": replicate,
                "library_name": library_name,
                "platform_id": platform,
                "instrument_model": instrument,
                "sra_experiment": experiment,
                "sra_run": run,
                "sra_spots": spots,
                "sra_runinfo_url": f"https://trace.ncbi.nlm.nih.gov/Traces/sra-db-be/runinfo?acc={experiment}",
                "official_geo_record": f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={accession}",
                "candidate_series_cooler": cooler_id,
                "cooler_contact_sum": cooler_sum,
                "cooler_sum_over_sra_spots": ratio,
                "mapping_assessment": assessment,
                "cooler_file_mapping_confirmed": False,
            }
        )
    audit = pd.DataFrame(rows)
    audit.to_csv(OUT / "44c_geo_sample_platform_audit.csv", index=False)
    return audit


def verify_series_coolers(sample_table: pd.DataFrame) -> None:
    rows = []
    for sample in sample_table[sample_table.condition.isin(["37C", "44C"])].itertuples():
        path = Path(sample.path)
        if not path.is_file():
            raise FileNotFoundError(path)
        matrix = cooler.Cooler(str(path))
        observed_sum = int(matrix.info["sum"])
        if observed_sum != int(sample.library_pairs):
            raise ValueError(f"Cooler sum mismatch for {sample.sample_id}: {observed_sum}")
        if matrix.binsize != 10 or matrix.chromsizes.to_dict() != {"NC_000913.3": GENOME_BP}:
            raise ValueError(f"Unexpected Cooler axes for {sample.sample_id}")
        rows.append(
            {
                "sample_id": sample.sample_id,
                "condition": sample.condition,
                "replicate": int(sample.replicate),
                "path": sample.path,
                "cooler_sum": observed_sum,
                "binsize": matrix.binsize,
                "chromosome": "NC_000913.3",
                "chromosome_length": GENOME_BP,
                "metadata_check": "PASS",
            }
        )
    pd.DataFrame(rows).to_csv(OUT / "series_cooler_integrity.csv", index=False)


def write_platform_comparability() -> None:
    rows = [
        {
            "condition": "37C",
            "sample_accessions": "GSM8394774;GSM8394776 (inferred t2)",
            "official_geo_records": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM8394774;https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM8394776",
            "platform_id": "GPL34700",
            "instrument_model": "DNBSEQ-T7",
            "cooler_source": "GSE272159 series-level rep1/rep2; inferred from t2 labels and SRA spot-count scale",
            "comparison_to_wt": "reference",
            "interpretation_limit": "GSM-to-Cooler mapping is not explicitly recorded",
        },
        {
            "condition": "44C",
            "sample_accessions": "GSM8394775;GSM8394777 (inferred t2)",
            "official_geo_records": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM8394775;https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM8394777",
            "platform_id": "GPL34700",
            "instrument_model": "DNBSEQ-T7",
            "cooler_source": "GSE272159 series-level rep1/rep2; inferred from t2 labels and SRA spot-count scale",
            "comparison_to_wt": "platform-matched under the t2 inference",
            "interpretation_limit": "GSM-to-Cooler mapping is not explicitly recorded",
        },
        {
            "condition": "FIS_mut",
            "sample_accessions": "GSM9078280;GSM9078281",
            "official_geo_records": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM9078280;https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM9078281",
            "platform_id": "GPL35987",
            "instrument_model": "DNBSEQ-G50",
            "cooler_source": "GSM-specific supplementary Cooler files",
            "comparison_to_wt": "different instrument from inferred T7 WT",
            "interpretation_limit": "condition and platform are not separated by a same-platform WT control",
        },
        {
            "condition": "HNS_mut",
            "sample_accessions": "GSM9678478;GSM9678479",
            "official_geo_records": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM9678478;https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM9678479",
            "platform_id": "GPL33918",
            "instrument_model": "DNBSEQ-G400",
            "cooler_source": "GSM-specific supplementary Cooler files",
            "comparison_to_wt": "different instrument from inferred T7 WT",
            "interpretation_limit": "condition and platform are not separated by a same-platform WT control",
        },
    ]
    pd.DataFrame(rows).to_csv(OUT / "condition_platform_comparability.csv", index=False)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    features = pd.read_csv(STRUCTURES)
    samples = pd.read_csv(SAMPLES)
    if len(features) != 344:
        raise ValueError(f"Expected 344 structures, found {len(features)}")

    lengths = (features.end - features.start).to_numpy(dtype=int)
    extended = features[["chrom"]].copy()
    extended["start"] = ((features.start - 2 * lengths) // 200) * 200
    extended["end"] = ((features.end + 2 * lengths + 199) // 200) * 200
    features["context_group"] = overlap_groups(extended).to_numpy()
    group_sizes = features.groupby("context_group").size().to_dict()
    all_annotated = np.zeros(GENOME_BINS, dtype=bool)
    for feature in features.itertuples():
        lo = max(0, int(feature.start) // BIN_BP)
        hi = min(len(all_annotated), int(np.ceil(feature.end / BIN_BP)))
        all_annotated[lo:hi] = True

    candidate_table = load_candidate_tables()
    all_samples = samples[samples.condition.isin(["37C", "44C", "HNS_mut", "FIS_mut"])].copy()
    cache_handles: dict[str, object] = {}
    for sample in all_samples.itertuples():
        cache_root = Path(
            "data/processed/structure_conditions/44c_double_flank"
            if sample.condition in {"37C", "44C"}
            else "data/processed/structure_conditions/double_flank"
        )
        cache_path = cache_root / f"{sample.sample_id}.npz"
        if not cache_path.is_file():
            raise FileNotFoundError(cache_path)
        cache_handles[sample.sample_id] = np.load(cache_path)

    rows = []
    try:
        for candidate in candidate_table.itertuples():
            structure_id = candidate.structure_id
            condition = candidate.condition
            feature = features.loc[features.structure_id == structure_id].iloc[0]
            components = summarize_components(
                structure_id, condition, features, all_samples, cache_handles, all_annotated
            )
            local_effect = float(candidate.w1_clean_effect)
            target_effect = float(components["target_log2_change"])
            background_effect = float(components["background_log2_change"])
            error = local_effect - (target_effect - background_effect)
            if abs(error) > 2e-5:
                raise ValueError(f"Decomposition mismatch for {structure_id}/{condition}: {error}")

            direct_overlaps = features.loc[
                (features.start < feature.end)
                & (features.end > feature.start)
                & (features.structure_id != structure_id),
                "structure_id",
            ].astype(str).tolist()
            target_same = np.sign(target_effect) == np.sign(local_effect)
            target_qualifies = (
                bool(components["target_all_four_same_direction"])
                and target_same
                and abs(target_effect) >= MIN_TARGET_CHANGE
            )
            rows.append(
                {
                    "condition": condition,
                    "structure_id": structure_id,
                    "type": feature.type,
                    "chrom": feature.chrom,
                    "start": int(feature.start),
                    "end": int(feature.end),
                    "context_group": int(feature.context_group),
                    "context_group_annotation_count": int(group_sizes[int(feature.context_group)]),
                    "other_annotated_intervals_overlapping_target": ";".join(direct_overlaps),
                    "direct_interval_overlap_count": len(direct_overlaps),
                    "local_enrichment_log2_change": local_effect,
                    "local_enrichment_fold_change": float(2**local_effect),
                    "target_density_log2_change": target_effect,
                    "target_density_fold_change": float(2**target_effect),
                    "target_density_all_four_cross_pairs_same_direction": bool(components["target_all_four_same_direction"]),
                    "target_density_minimum_cross_pair_abs_log2": float(components["target_minimum_cross_effect"]),
                    "background_density_log2_change": background_effect,
                    "background_density_fold_change": float(2**background_effect),
                    "background_density_all_four_cross_pairs_same_direction": bool(components["background_all_four_same_direction"]),
                    "target_change_supports_local_change_ge_0_25_log2": bool(target_qualifies),
                    "evidence_interpretation": interpretation(
                        local_effect,
                        target_effect,
                        background_effect,
                        bool(components["target_all_four_same_direction"]),
                    ),
                    "local_decomposition_absolute_error": abs(error),
                    "robust_candidate_rule_passed": True,
                    **components,
                }
            )
    finally:
        for cache in cache_handles.values():
            cache.close()

    combined = pd.DataFrame(rows).sort_values(["condition", "start", "structure_id"]).reset_index(drop=True)
    if len(combined) != 26:
        raise ValueError(f"Expected 26 retained comparisons, found {len(combined)}")
    combined.to_csv(OUT / "combined_robust_candidates.csv", index=False)

    context_rows = []
    for group_id, group in combined.groupby("context_group", sort=True):
        context_rows.append(
            {
                "context_group": int(group_id),
                "annotated_structures_in_group": int(group_sizes[int(group_id)]),
                "candidate_comparisons": int(len(group)),
                "conditions": ";".join(sorted(group.condition.unique())),
                "candidate_structures": ";".join(sorted(group.structure_id.unique())),
                "comparison_ids": ";".join(
                    f"{row.structure_id}:{row.condition}" for row in group.itertuples()
                ),
                "directly_overlapping_candidate_intervals": int(group.direct_interval_overlap_count.sum()),
            }
        )
    context_summary = pd.DataFrame(context_rows)
    context_summary.to_csv(OUT / "candidate_context_groups.csv", index=False)

    pair_rows = []
    records = list(combined.itertuples())
    for index, first in enumerate(records):
        for second in records[index + 1 :]:
            overlap_bp = max(0, min(first.end, second.end) - max(first.start, second.start))
            same_annotation = first.structure_id == second.structure_id
            same_context = first.context_group == second.context_group
            if overlap_bp or same_annotation or same_context:
                if same_annotation:
                    relationship = "same annotated interval tested in multiple conditions"
                elif overlap_bp:
                    relationship = "distinct candidate intervals overlap in base coordinates"
                else:
                    relationship = "distinct annotations share the expanded context group"
                pair_rows.append(
                    {
                        "structure_a": first.structure_id,
                        "condition_a": first.condition,
                        "structure_b": second.structure_id,
                        "condition_b": second.condition,
                        "context_group_a": int(first.context_group),
                        "context_group_b": int(second.context_group),
                        "overlap_bp": int(overlap_bp),
                        "same_annotation": bool(same_annotation),
                        "same_expanded_context_group": bool(same_context),
                        "relationship": relationship,
                    }
                )
    pd.DataFrame(pair_rows).to_csv(OUT / "candidate_pair_relationships.csv", index=False)

    by_condition = {}
    for condition, group in combined.groupby("condition", sort=True):
        by_condition[condition] = {
            "comparisons": int(len(group)),
            "unique_structures": int(group.structure_id.nunique()),
            "context_groups": int(group.context_group.nunique()),
            "target_directionally_consistent_all_four_pairs": int(
                group.target_density_all_four_cross_pairs_same_direction.sum()
            ),
            "target_support_ge_0_25_log2_and_same_direction": int(
                group.target_change_supports_local_change_ge_0_25_log2.sum()
            ),
            "target_centered": int(group.evidence_interpretation.str.startswith("target-centered").sum()),
            "flank_dominant_or_background_driven": int(
                group.evidence_interpretation.str.startswith(("flank-dominant", "background-driven")).sum()
            ),
            "mixed_with_larger_flank_contribution": int(
                group.evidence_interpretation.str.startswith("mixed,").sum()
            ),
        }

    annotation_ids = sorted(combined.structure_id.unique())
    repeated_structures = {
        structure_id: sorted(group.condition.unique())
        for structure_id, group in combined.groupby("structure_id")
        if group.condition.nunique() > 1
    }
    summary = {
        "comparison_count": int(len(combined)),
        "unique_annotation_count": int(combined.structure_id.nunique()),
        "unique_candidate_context_groups": int(combined.context_group.nunique()),
        "context_groups_with_multiple_comparison_rows": int((context_summary.candidate_comparisons > 1).sum()),
        "candidate_comparison_rows_overlapping_other_known_intervals": int(
            (combined.direct_interval_overlap_count > 0).sum()
        ),
        "candidate_pair_relationship_count": int(len(pair_rows)),
        "distinct_candidate_interval_overlap_pairs": int(
            sum(row["overlap_bp"] > 0 and not row["same_annotation"] for row in pair_rows)
        ),
        "condition_comparisons": by_condition,
        "structures_repeated_across_conditions": repeated_structures,
        "target_threshold_log2": MIN_TARGET_CHANGE,
        "decomposition_check": {
            "identity": "local enrichment log2 change = target density log2 change - flank density log2 change",
            "maximum_absolute_error": float(combined.local_decomposition_absolute_error.max()),
            "all_rows_within_2e-5": bool((combined.local_decomposition_absolute_error <= 2e-5).all()),
        },
        "independence_note": "Condition-structure comparisons are not independent discoveries; annotations can share expanded genomic context and conditions share WT matrices.",
        "sample_mapping_note": "44C and WT series-level Coolers are strongly consistent with the t2 DNBSEQ-T7 replicates by SRA spot-count scale and replicate labels, but the series files do not provide explicit GSM links.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    sample_audit = write_geo_mapping_audit(samples)
    verify_series_coolers(samples)
    write_platform_comparability()
    print("Condition summary:")
    print(pd.DataFrame(by_condition).T.to_string())
    print("\nCombined counts:", summary["comparison_count"], "rows;", summary["unique_annotation_count"], "annotations;", summary["unique_candidate_context_groups"], "context groups")
    print("Repeated structures:", repeated_structures)
    print("44C matrix-to-SRA ratios by timepoint:")
    print(sample_audit.groupby(["condition", "timepoint_group"]).cooler_sum_over_sra_spots.agg(["min", "max"]).to_string())
    print("Maximum decomposition error:", summary["decomposition_check"]["maximum_absolute_error"])


if __name__ == "__main__":
    main()

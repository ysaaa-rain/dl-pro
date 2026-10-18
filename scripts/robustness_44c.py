"""Run the fixed structure-condition robustness screen for WT versus 44C."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from dlpro.data import overlap_groups
from dlpro.experiment_core import ContactReader, save_json


ROOT = Path("results/structure_conditions/44C")
CACHE = Path("data/processed/structure_conditions/44c_double_flank")
SAMPLE_TABLE = Path("data/annotations/samples.csv")
STRUCTURE_TABLE = Path("data/annotations/structures.csv")
WIDTHS = (0.5, 1.0, 2.0)
BACKGROUNDS = ("clean", "all")
BIN_BP = 100
MIN_DISTANCE_BP = 200
MAX_DISTANCE_FACTOR = 0.5
MIN_EFFECT = 0.25


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_inputs(samples: pd.DataFrame) -> None:
    if len(samples) != 4 or set(samples.condition) != {"37C", "44C"}:
        raise ValueError("Expected exactly two WT and two 44C Cooler matrices")
    for condition in ("37C", "44C"):
        rows = samples[samples.condition == condition]
        if sorted(rows.replicate.astype(int).tolist()) != [1, 2]:
            raise ValueError(f"{condition} must have biological replicates 1 and 2")
    for row in samples.itertuples():
        path = Path(row.path)
        if not path.is_file():
            raise FileNotFoundError(path)
        if int(path.stat().st_size) != int(row.bytes):
            raise ValueError(f"Byte-size mismatch for {path}")
        if sha256(path) != row.sha256:
            raise ValueError(f"SHA-256 mismatch for {path}")
        if int(row.binsize) != 10 or row.chrom != "NC_000913.3":
            raise ValueError(f"Unexpected reference metadata for {path}")
        if int(row.chromosome_length) != 4_641_652:
            raise ValueError(f"Unexpected chromosome length for {path}")


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    features = pd.read_csv(STRUCTURE_TABLE)
    samples = pd.read_csv(SAMPLE_TABLE)
    samples = samples[samples.condition.isin(["37C", "44C"])].copy()
    samples = samples.sort_values(["condition", "replicate"]).reset_index(drop=True)
    if len(features) != 344:
        raise ValueError(f"Expected 344 annotated structures, found {len(features)}")
    ensure_inputs(samples)

    lengths = (features.end - features.start).to_numpy(dtype=int)
    ext = features[["chrom"]].copy()
    ext["start"] = ((features.start - 2 * lengths) // 200) * 200
    ext["end"] = ((features.end + 2 * lengths + 199) // 200) * 200
    features["context_group"] = overlap_groups(ext).to_numpy()
    group_count = int(features.context_group.nunique())

    all_annotated = np.zeros(464_166, dtype=bool)
    for feature in features.itertuples():
        lo = max(0, int(feature.start) // BIN_BP)
        hi = min(len(all_annotated), int(np.ceil(feature.end / BIN_BP)))
        all_annotated[lo:hi] = True

    # Cache each structure's 2x-flank window to support all six fixed sensitivities.
    expanded: dict[str, np.lib.npyio.NpzFile] = {}
    source_samples = []
    for sample in samples.itertuples():
        cache_path = CACHE / f"{sample.sample_id}.npz"
        if not cache_path.exists():
            reader = ContactReader(sample.path)
            matrices = {}
            for index, feature in enumerate(features.itertuples()):
                length = int(feature.end - feature.start)
                lo = ((int(feature.start) - 2 * length) // BIN_BP) * BIN_BP
                hi = int(np.ceil((int(feature.end) + 2 * length) / BIN_BP)) * BIN_BP
                if lo < 0 or hi > 4_641_652:
                    raise ValueError(f"Expanded window crosses chromosome edge: {feature.structure_id}")
                matrices[feature.structure_id] = reader.window(lo // BIN_BP, (hi - lo) // BIN_BP, BIN_BP)
                if index % 100 == 0:
                    print(f"{sample.sample_id}: extracted {index + 1}/344 windows", flush=True)
            np.savez_compressed(cache_path, **matrices)
        expanded[sample.sample_id] = np.load(cache_path)
        source_samples.append(
            {
                "sample_id": sample.sample_id,
                "condition": sample.condition,
                "replicate": int(sample.replicate),
                "path": sample.path,
                "sha256": sample.sha256,
                "bytes": int(sample.bytes),
                "library_pairs": int(sample.library_pairs),
                "cooler_sha256_verified": True,
            }
        )

    metrics: dict[tuple[str, str, float, str], tuple[float, float, float, int, int]] = {}
    for feature in features.itertuples():
        length = int(feature.end - feature.start)
        for sample in samples.itertuples():
            matrix = expanded[sample.sample_id][feature.structure_id]
            lo = ((int(feature.start) - 2 * length) // BIN_BP) * BIN_BP
            centers = lo + (np.arange(len(matrix)) + 0.5) * BIN_BP
            core = (centers >= feature.start) & (centers < feature.end)
            other_annotations = all_annotated[(centers // BIN_BP).astype(int)]
            for multiplier in WIDTHS:
                flank_width = multiplier * length
                left = (centers >= feature.start - flank_width) & (centers < feature.start)
                right = (centers >= feature.end) & (centers < feature.end + flank_width)
                flank = left | right
                clean_flank = flank & ~other_annotations
                max_distance = max(
                    2,
                    int(np.floor(min(flank_width, MAX_DISTANCE_FACTOR * length) / BIN_BP)),
                )
                for background_name, background in (("all", flank), ("clean", clean_flank)):
                    target_total = 0.0
                    background_scaled = 0.0
                    target_pixels = 0
                    for distance in range(MIN_DISTANCE_BP // BIN_BP, max_distance + 1):
                        if distance >= len(matrix):
                            break
                        values = np.diag(matrix, distance)
                        target_pairs = core[:-distance] & core[distance:]
                        background_pairs = background[:-distance] & background[distance:]
                        if target_pairs.any() and background_pairs.any():
                            n_target = int(target_pairs.sum())
                            mean_background = float(values[background_pairs].mean())
                            if mean_background > 0:
                                target_total += float(values[target_pairs].sum())
                                background_scaled += mean_background * n_target
                                target_pixels += n_target
                    enrichment = (
                        target_total / background_scaled
                        if target_total > 0 and background_scaled > 0
                        else np.nan
                    )
                    target_density = target_total / target_pixels if target_pixels else np.nan
                    background_density = background_scaled / target_pixels if target_pixels else np.nan
                    metrics[(feature.structure_id, sample.sample_id, multiplier, background_name)] = (
                        enrichment,
                        target_density,
                        background_density,
                        target_pixels,
                        int(clean_flank.sum()),
                    )

    rows = []
    within = []
    for feature in features.itertuples():
        row = {
            "structure_id": feature.structure_id,
            "type": feature.type,
            "chrom": feature.chrom,
            "start": int(feature.start),
            "end": int(feature.end),
            "context_group": int(feature.context_group),
            "condition": "44C",
        }
        for multiplier in WIDTHS:
            for background in BACKGROUNDS:
                tag = f"w{multiplier:g}_{background}"
                wt_ids = samples[(samples.condition == "37C")].sort_values("replicate").sample_id.tolist()
                heat_ids = samples[(samples.condition == "44C")].sort_values("replicate").sample_id.tolist()
                sample_values = {
                    sid: metrics[(feature.structure_id, sid, multiplier, background)]
                    for sid in wt_ids + heat_ids
                }
                wt = np.array([sample_values[sid][0] for sid in wt_ids])
                heat = np.array([sample_values[sid][0] for sid in heat_ids])
                wt_target = np.array([sample_values[sid][1] for sid in wt_ids])
                heat_target = np.array([sample_values[sid][1] for sid in heat_ids])
                wt_background = np.array([sample_values[sid][2] for sid in wt_ids])
                heat_background = np.array([sample_values[sid][2] for sid in heat_ids])
                valid = np.all(np.isfinite(wt)) and np.all(np.isfinite(heat)) and np.all(wt > 0) and np.all(heat > 0)
                if valid:
                    wt_log = np.log2(wt)
                    heat_log = np.log2(heat)
                    cross = (heat_log[:, None] - wt_log[None, :]).ravel()
                    effect = float(heat_log.mean() - wt_log.mean())
                    same_direction = bool(np.all(cross > 0) or np.all(cross < 0))
                    minimum_cross = float(np.min(np.abs(cross)))
                    wt_repeat = float(abs(wt_log[1] - wt_log[0]))
                    heat_repeat = float(abs(heat_log[1] - heat_log[0]))
                    absolute_changes = {}
                    for name, wt_values, heat_values in (
                        ("target", wt_target, heat_target),
                        ("background", wt_background, heat_background),
                    ):
                        if np.all(np.isfinite(wt_values)) and np.all(np.isfinite(heat_values)) and np.all(wt_values > 0) and np.all(heat_values > 0):
                            wt_density_log = np.log2(wt_values)
                            heat_density_log = np.log2(heat_values)
                            density_cross = (heat_density_log[:, None] - wt_density_log[None, :]).ravel()
                            absolute_changes[name] = (
                                float(heat_density_log.mean() - wt_density_log.mean()),
                                bool(np.all(density_cross > 0) or np.all(density_cross < 0)),
                                float(np.min(np.abs(density_cross))),
                            )
                        else:
                            absolute_changes[name] = (np.nan, False, np.nan)
                else:
                    effect = np.nan
                    same_direction = False
                    minimum_cross = np.nan
                    wt_repeat = np.nan
                    heat_repeat = np.nan
                    absolute_changes = {name: (np.nan, False, np.nan) for name in ("target", "background")}
                row.update(
                    {
                        f"{tag}_effect": effect,
                        f"{tag}_all_four_same_direction": same_direction,
                        f"{tag}_minimum_cross_effect": minimum_cross,
                        f"{tag}_WT_repeat_difference": wt_repeat,
                        f"{tag}_44C_repeat_difference": heat_repeat,
                        f"{tag}_target_density_log2_change": absolute_changes["target"][0],
                        f"{tag}_target_density_all_four_same_direction": absolute_changes["target"][1],
                        f"{tag}_target_density_minimum_cross_effect": absolute_changes["target"][2],
                        f"{tag}_background_density_log2_change": absolute_changes["background"][0],
                        f"{tag}_background_density_all_four_same_direction": absolute_changes["background"][1],
                        f"{tag}_background_density_minimum_cross_effect": absolute_changes["background"][2],
                    }
                )
                for sid in wt_ids + heat_ids:
                    row[f"{tag}_{sid}_enrichment"] = float(sample_values[sid][0]) if np.isfinite(sample_values[sid][0]) else np.nan
                    row[f"{tag}_{sid}_target_density"] = float(sample_values[sid][1]) if np.isfinite(sample_values[sid][1]) else np.nan
                    row[f"{tag}_{sid}_background_density"] = float(sample_values[sid][2]) if np.isfinite(sample_values[sid][2]) else np.nan
                for condition, difference in (("37C", wt_repeat), ("44C", heat_repeat)):
                    if np.isfinite(difference):
                        within.append(
                            {
                                "structure_id": feature.structure_id,
                                "context_group": int(feature.context_group),
                                "condition": condition,
                                "flank_width": multiplier,
                                "background": background,
                                "abs_repeat_log2": difference,
                            }
                        )
        rows.append(row)

    result = pd.DataFrame(rows)
    repeat_noise = pd.DataFrame(within)
    null_rows = []
    for multiplier in WIDTHS:
        for background in BACKGROUNDS:
            group_maxima = (
                repeat_noise[(repeat_noise.flank_width == multiplier) & (repeat_noise.background == background)]
                .groupby(["condition", "context_group"])
                .abs_repeat_log2.max()
            )
            null = group_maxima.to_numpy()
            q95 = float(np.quantile(null, 0.95))
            q99 = float(np.quantile(null, 0.99))
            largest = float(null.max())
            effect_col = f"w{multiplier:g}_{background}_effect"
            absolute_effect = result[effect_col].abs()
            valid_effects = absolute_effect[np.isfinite(absolute_effect)]
            null_rows.append(
                {
                    "condition": "44C",
                    "flank_width": multiplier,
                    "background": background,
                    "empirical_null_groups": int(len(null)),
                    "within_repeat_q95": q95,
                    "within_repeat_q99": q99,
                    "largest_repeat_noise": largest,
                    "valid_comparisons": int(len(valid_effects)),
                    "above_q99": int((valid_effects > q99).sum()),
                    "above_largest_repeat_noise": int((valid_effects > largest).sum()),
                }
            )
    null_summary = pd.DataFrame(null_rows)

    sensitivity_robust = pd.Series(True, index=result.index)
    noise_robust = pd.Series(True, index=result.index)
    largest_noise_robust = pd.Series(True, index=result.index)
    for multiplier in WIDTHS:
        for background in BACKGROUNDS:
            prefix = f"w{multiplier:g}_{background}"
            sensitivity_robust &= (
                result[f"{prefix}_effect"].abs().ge(MIN_EFFECT)
                & result[f"{prefix}_all_four_same_direction"]
                & result[f"{prefix}_minimum_cross_effect"].ge(MIN_EFFECT)
            )
            cutoff = null_summary.loc[
                (null_summary.flank_width == multiplier) & (null_summary.background == background),
                "within_repeat_q99",
            ].iloc[0]
            maximum = null_summary.loc[
                (null_summary.flank_width == multiplier) & (null_summary.background == background),
                "largest_repeat_noise",
            ].iloc[0]
            noise_robust &= result[f"{prefix}_effect"].abs().gt(cutoff)
            largest_noise_robust &= result[f"{prefix}_effect"].abs().gt(maximum)
    result["all_six_sensitivities_directional"] = sensitivity_robust
    result["exceeds_repeat_noise_q99_all_sensitivities"] = noise_robust
    result["exceeds_largest_repeat_difference_all_sensitivities"] = largest_noise_robust
    result["robust_candidate"] = sensitivity_robust & noise_robust & largest_noise_robust

    out = ROOT / "robustness"
    out.mkdir(parents=True, exist_ok=True)
    result.to_csv(out / "all_structure_sensitivities.csv", index=False)
    repeat_noise.to_csv(out / "within_condition_repeat_noise.csv", index=False)
    null_summary.to_csv(out / "empirical_noise_summary.csv", index=False)
    candidates = result[result.robust_candidate].copy()
    candidates["abs_w1_clean_effect"] = candidates.w1_clean_effect.abs()
    candidates = candidates.sort_values("abs_w1_clean_effect", ascending=False)
    candidates.to_csv(out / "robust_candidates.csv", index=False)
    decomposition = candidates[
        [
            "structure_id",
            "type",
            "context_group",
            "w1_clean_effect",
            "w1_clean_target_density_log2_change",
            "w1_clean_target_density_all_four_same_direction",
            "w1_clean_target_density_minimum_cross_effect",
            "w1_clean_background_density_log2_change",
            "w1_clean_background_density_all_four_same_direction",
            "robust_candidate",
        ]
    ].copy()
    labels = []
    for row in decomposition.itertuples():
        target_change = row.w1_clean_target_density_log2_change
        effect = row.w1_clean_effect
        if (
            row.w1_clean_target_density_all_four_same_direction
            and abs(target_change) >= MIN_EFFECT
            and np.sign(target_change) == np.sign(effect)
        ):
            labels.append("target contact density changes in the same direction as local enrichment")
        elif abs(target_change) < MIN_EFFECT or np.sign(target_change) != np.sign(effect):
            labels.append("relative enrichment shift is primarily background-mediated")
        else:
            labels.append("target contact change is not consistent across all replicate pairings")
    decomposition["interpretation"] = labels
    decomposition.to_csv(out / "robust_candidate_decomposition.csv", index=False)

    geo_platform_rows = []
    for timepoint, platform, instrument, wt_ids, heat_ids in (
        ("t1", "GPL32081", "NextSeq 2000", ("GSM8394770", "GSM8394772"), ("GSM8394771", "GSM8394773")),
        ("t2", "GPL34700", "DNBSEQ-T7", ("GSM8394774", "GSM8394776"), ("GSM8394775", "GSM8394777")),
    ):
        for condition, accessions in (("37C", wt_ids), ("44C", heat_ids)):
            for replicate, accession in enumerate(accessions, start=1):
                geo_platform_rows.append(
                    {
                        "sample_accession": accession,
                        "condition": condition,
                        "timepoint_group": timepoint,
                        "replicate": replicate,
                        "platform_id": platform,
                        "instrument_model": instrument,
                        "official_record": f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={accession}",
                        "cooler_file_mapping_confirmed": False,
                    }
                )
    pd.DataFrame(geo_platform_rows).to_csv(out / "geo_sample_platform_audit.csv", index=False)

    top_rows = result.assign(abs_w1_clean_effect=result.w1_clean_effect.abs()).sort_values(
        "abs_w1_clean_effect", ascending=False
    )
    to_plot = candidates if len(candidates) else top_rows.head(3)
    heatmap_rows = []
    for feature in to_plot.itertuples():
        structure = features.loc[features.structure_id == feature.structure_id].iloc[0]
        figure_samples = samples.sort_values(["condition", "replicate"])
        images = []
        lo_bp = ((int(structure.start) - 2 * int(structure.end - structure.start)) // BIN_BP) * BIN_BP
        for sample in figure_samples.itertuples():
            matrix = expanded[sample.sample_id][feature.structure_id]
            images.append(np.log1p(matrix))
        vmax = float(np.quantile(np.concatenate([image.ravel() for image in images]), 0.99))
        fig, axes = plt.subplots(2, 2, figsize=(9, 7))
        for axis, sample, image in zip(axes.flat, figure_samples.itertuples(), images):
            axis.imshow(
                image,
                origin="lower",
                cmap="magma",
                vmin=0,
                vmax=vmax,
                extent=[lo_bp / 1000, (lo_bp + len(image) * BIN_BP) / 1000] * 2,
            )
            axis.axvspan(structure.start / 1000, structure.end / 1000, color="cyan", alpha=0.08)
            axis.axvline(structure.start / 1000, color="cyan", lw=0.8)
            axis.axvline(structure.end / 1000, color="cyan", lw=0.8)
            axis.axhline(structure.start / 1000, color="cyan", lw=0.8)
            axis.axhline(structure.end / 1000, color="cyan", lw=0.8)
            axis.set_title(f"{sample.condition} rep{int(sample.replicate)}")
            axis.set_xlabel("Genome coordinate (kb)")
        suffix = "candidate" if bool(feature.robust_candidate) else "descriptive_top"
        fig.suptitle(
            f"{feature.structure_id}, 44C vs WT: local enrichment log2 change "
            f"{feature.w1_clean_effect:+.2f} ({suffix})"
        )
        fig.tight_layout()
        image_path = out / f"{feature.structure_id}_{suffix}.png"
        fig.savefig(image_path, dpi=150)
        plt.close(fig)
        heatmap_rows.append(
            {
                "structure_id": feature.structure_id,
                "robust_candidate": bool(feature.robust_candidate),
                "w1_clean_effect": float(feature.w1_clean_effect),
                "heatmap": image_path.name,
            }
        )
    pd.DataFrame(heatmap_rows).to_csv(out / "heatmap_index.csv", index=False)
    thumbnails = []
    for item in heatmap_rows:
        image = Image.open(out / item["heatmap"]).convert("RGB")
        image.thumbnail((480, 360))
        tile = Image.new("RGB", (500, 400), "white")
        tile.paste(image, ((500 - image.width) // 2, 0))
        ImageDraw.Draw(tile).text((8, 370), item["structure_id"], fill="black")
        thumbnails.append(tile)
    if thumbnails:
        columns = 4
        rows_count = (len(thumbnails) + columns - 1) // columns
        montage = Image.new("RGB", (columns * 500, rows_count * 400), "#dddddd")
        for index, image in enumerate(thumbnails):
            montage.paste(image, ((index % columns) * 500, (index // columns) * 400))
        montage.save(out / "candidate_montage.png")

    candidate_summary = {
        "condition": "44C",
        "comparison": "44C versus 37C WT",
        "structures_screened": int(len(result)),
        "context_groups": group_count,
        "robust_candidates": int(result.robust_candidate.sum()),
        "directional_across_all_sensitivities": int(sensitivity_robust.sum()),
        "exceed_noise_q99_across_all_sensitivities": int(noise_robust.sum()),
        "exceed_maximum_repeat_noise_across_all_sensitivities": int(largest_noise_robust.sum()),
        "top_descriptive_structures": top_rows.head(5)[
            ["structure_id", "type", "w1_clean_effect"]
        ].to_dict(orient="records"),
        "robust_candidates_with_target_density_change_in_same_direction": int(
            decomposition.interpretation.eq(
                "target contact density changes in the same direction as local enrichment"
            ).sum()
        ),
        "robust_candidates_with_background_mediated_relative_shift": int(
            decomposition.interpretation.eq(
                "relative enrichment shift is primarily background-mediated"
            ).sum()
        ),
    }
    null_at_primary = null_summary[
        (null_summary.flank_width == 1.0) & (null_summary.background == "clean")
    ].iloc[0]
    w1 = result.w1_clean_effect.dropna().abs()
    repeat_ref = repeat_noise[
        (repeat_noise.flank_width == 1.0) & (repeat_noise.background == "clean")
    ].groupby(["condition", "context_group"]).abs_repeat_log2.max()
    inputs = {
        "source_samples": source_samples,
        "source_accession": "GSE272159",
        "source_urls": {
            row.sample_id: row.source_url for row in samples.itertuples()
        },
    }
    protocol = {
        "comparison": "44C heat shock versus 37C WT",
        "structures_screened": 344,
        "flank_width_multipliers": list(WIDTHS),
        "bin_bp": BIN_BP,
        "distance_support": {
            "minimum": f"{MIN_DISTANCE_BP} bp",
            "maximum": f"{MAX_DISTANCE_FACTOR} times annotated structure length",
            "same_across_flank_widths": True,
        },
        "backgrounds": {
            "all": "all bins in the left and right flanks",
            "clean": "flank bins not overlapping another known structure annotation",
        },
        "normalization": "per-million depth normalization in ContactReader, then target enrichment relative to distance-matched flank contact density",
        "replicate_design": "two unpaired biological replicates in each condition; evaluate all four cross-condition replicate contrasts",
        "screening_scope": "all 344 annotations; no prior condition-specific candidate list used",
        "minimum_effect_log2": MIN_EFFECT,
        "empirical_noise": "For each sensitivity, maximum absolute within-condition replicate log2 difference per largest-context overlap group; pooled across WT and 44C. Descriptive threshold, not p-values, FDR, or family-wise error control.",
        "candidate_rule": "Direction and at least 0.25 absolute log2 effect in all six flank/background settings, all four cross-condition replicate contrasts direction-concordant with minimum absolute contrast of at least 0.25, and every setting's effect larger than both its pooled group-noise 99th percentile and largest observed repeat-noise maximum.",
        "retrospective": True,
        "limitations": [
            "Two biological replicates per condition limit formal inference at individual structures.",
            "The empirical noise screen is descriptive and does not provide multiplicity-adjusted p-values or FDR control.",
            "The contact maps and annotated positions have appeared in prior project analyses; this is not an independent validation set.",
            "GEO lists heat-shock sample records across distinct t1 and t2 sequencing batches/platforms; the series-level 44C Cooler filenames do not identify a GSM accession. This screen can identify stable matrix-level differences but cannot alone rule out residual platform or batch effects.",
        ],
        "inputs": inputs,
    }
    save_json(out / "protocol.json", protocol)
    save_json(out / "summary.json", {**candidate_summary, "primary_sensitivity": {
        "background": "clean",
        "flank_width_multiplier": 1.0,
        "absolute_effect_q95": float(np.quantile(w1, 0.95)),
        "absolute_effect_q99": float(np.quantile(w1, 0.99)),
        "empirical_repeat_noise_q95": float(null_at_primary.within_repeat_q95),
        "empirical_repeat_noise_q99": float(null_at_primary.within_repeat_q99),
        "largest_group_repeat_noise": float(null_at_primary.largest_repeat_noise),
        "median_repeat_noise_max_per_context": float(repeat_ref.median()),
    }, "flank_multipliers": list(WIDTHS), "backgrounds": ["all", "clean"]})

    output_integrity = {}
    for path in sorted(out.iterdir()):
        if path.is_file() and path.name != "integrity.json":
            output_integrity[path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    save_json(
        out / "integrity.json",
        {
            "input_sample_count": len(source_samples),
            "input_hashes_verified": True,
            "structures": int(len(result)),
            "condition_comparisons": int(len(result)),
            "sensitivity_settings_per_structure": len(WIDTHS) * len(BACKGROUNDS),
            "finite_primary_enrichments": int(result.w1_clean_effect.notna().sum()),
            "robust_candidates": int(result.robust_candidate.sum()),
            "heatmaps": heatmap_rows,
            "outputs": output_integrity,
        },
    )
    print(json.dumps(candidate_summary, ensure_ascii=False, indent=2))
    print("Primary noise reference:", json.dumps(null_at_primary.to_dict(), ensure_ascii=False))


if __name__ == "__main__":
    main()

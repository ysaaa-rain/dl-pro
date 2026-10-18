# 44C heat-shock structure comparison

This analysis compares the two series-level 44C Cooler matrices with the two 37C WT matrices across all 344 annotated structures. The four matrices have matching `NC_000913.3` axes, 10 bp native bins, the same Cooler format, and verified local SHA-256 values. Their Cooler sums are similar: WT 368.70M and 358.45M; 44C 376.33M and 352.93M. Contact windows are depth-normalized before local comparisons.

## Analysis

The target interval is compared with distance-matched contacts in its flanks. The analysis tests side widths of 0.5, 1, and 2 times the annotated feature length, with two background definitions: all flank bins, and flank bins not overlapping another known annotation. Every comparison uses the same target-distance support, from 200 bp through one half of the feature length. Both replicates are retained separately, and all four cross-condition replicate pairs are checked.

The empirical repeat-noise reference takes the maximum absolute within-condition replicate difference in each of the 81 overlapping-context groups, pooled across WT and 44C. At the primary one-length, clean-background setting this yields 162 group maxima; the 99th percentile is 0.240 log2 and the largest observed difference is 0.255 log2. A candidate must keep direction and an absolute effect of at least 0.25 log2 in all six sensitivity settings, agree in all four cross-replicate comparisons, and exceed both the setting-specific 99th-percentile and maximum repeat-noise values in every setting.

The screen found 16 robust local-enrichment changes among 323 structures with evaluable primary enrichment: eight increases and eight decreases, across 14 overlapping-context groups. They include 13 OPCIDs and 3 CHINs; no CHID passed. In a separate depth-normalized decomposition, 13 candidates had target contact density change in the same direction as local enrichment, while 3 were primarily background-mediated. These are matrix-level observations, not formal per-structure significance results.

| Structure | Local-enrichment change, log2 | 44C/WT enrichment ratio | Target-density change, log2 | Flank-density change, log2 | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| OPCID_65 | +1.299 | 2.46 | −0.154 | −1.452 | Relative enrichment rises mainly because the flank signal drops; the target itself does not rise. |
| OPCID_56 | −1.224 | 0.43 | −1.176 | +0.048 | Target contact density falls while flank density is nearly unchanged. |
| OPCID_53 | −0.999 | 0.50 | −1.332 | −0.334 | Target and flank both fall, with the larger change in the target. |
| OPCID_60 | +0.888 | 1.85 | +0.834 | −0.054 | Target contact density rises while flank density is nearly unchanged. |
| OPCID_45 | +0.766 | 1.70 | +0.816 | +0.050 | Target contact density rises while flank density is nearly unchanged. |
| CHIN_79 | −0.449 | 0.73 | −0.144 | +0.305 | Relative enrichment falls mainly because the flank signal rises. |

The decomposition prevents a relative enrichment change from being described automatically as absolute target strengthening or weakening. For example, OPCID_65 has the largest positive local-enrichment change, but its target density is nearly stable and its flanking density drops substantially. OPCID_60 is a clearer target-centered increase. CHIN_226 has a large clean-background point estimate (+0.871 log2 at one flank length) but does not pass the all-background, all-width rule, so it is excluded from the robust list.

## Data comparability and limits

The series-level Cooler files have condition-and-replicate names, but do not identify a specific GSM accession. NCBI's sample records list heat-shock and WT samples in two groups: t1 on NextSeq 2000 and t2 on DNBSEQ-T7. The exact mapping between those GSM records and the downloaded series-level Cooler matrices is not specified in the local series-file names; the platform audit is in `robustness/geo_sample_platform_audit.csv`. Thus, matching reference, resolution, format, and matrix sums do not rule out remaining sequencing-platform or batch effects. The observed differences should remain exploratory until this mapping is resolved or they are corroborated in a technically matched sample set.

Only two matrices are available per condition, so individual-structure inference is weak. The empirical noise thresholds are descriptive and are not permutation p-values, family-wise error control, or FDR. The same annotations and genomic positions were used in prior project analyses, so this is a retrospective robustness screen rather than independent validation.

## Files

- `robustness/all_structure_sensitivities.csv`: all 344 structure comparisons and six sensitivity settings.
- `robustness/robust_candidates.csv`: structures passing the frozen robustness rule.
- `robustness/robust_candidate_decomposition.csv`: target-versus-flank density changes for the robust candidates.
- `robustness/within_condition_repeat_noise.csv` and `robustness/empirical_noise_summary.csv`: repeat-noise reference values.
- `robustness/OPCID_65_candidate.png` and related images: matched 37C/44C repeat heatmaps with annotation boundaries.
- `robustness/candidate_montage.png`: overview of the retained candidates.
- `robustness/protocol.json` and `robustness/integrity.json`: fixed protocol, source hashes, and output checksums.

Source records: [GSE272159](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE272159), [WT t1 rep1](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM8394770), [heat-shock t1 rep1](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM8394771), [WT t2 rep1](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM8394774), and [heat-shock t2 rep1](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM8394775).

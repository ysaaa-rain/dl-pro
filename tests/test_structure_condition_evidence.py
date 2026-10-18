"""Consistency checks for the structure-condition evidence tables."""

import csv
import json
import math
from collections import Counter
from pathlib import Path


EVIDENCE = Path(__file__).resolve().parents[1] / "results" / "structure_conditions" / "evidence_synthesis"


def test_candidate_summary_and_density_decomposition():
    with (EVIDENCE / "combined_robust_candidates.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    summary = json.loads((EVIDENCE / "summary.json").read_text(encoding="utf-8"))

    assert rows
    assert len(rows) == summary["comparison_count"]
    assert len({row["structure_id"] for row in rows}) == summary["unique_annotation_count"]
    assert len({row["context_group"] for row in rows}) == summary["unique_candidate_context_groups"]
    assert len({(row["condition"], row["structure_id"]) for row in rows}) == len(rows)

    by_condition = Counter(row["condition"] for row in rows)
    assert set(by_condition) == set(summary["condition_comparisons"])
    for condition, count in by_condition.items():
        expected = summary["condition_comparisons"][condition]
        subset = [row for row in rows if row["condition"] == condition]
        assert count == expected["comparisons"]
        assert sum(row["evidence_interpretation"].startswith("target-centered") for row in subset) == expected["target_centered"]
        assert len({row["context_group"] for row in subset}) == expected["context_groups"]

    errors = []
    for row in rows:
        local = float(row["local_enrichment_log2_change"])
        target = float(row["target_density_log2_change"])
        background = float(row["background_density_log2_change"])
        assert all(math.isfinite(v) for v in (local, target, background))
        error = abs(local - (target - background))
        errors.append(error)
        assert error < 1e-10
        assert row["robust_candidate_rule_passed"] == "True"
    assert math.isclose(
        max(errors),
        summary["decomposition_check"]["maximum_absolute_error"],
        abs_tol=1e-12,
    )

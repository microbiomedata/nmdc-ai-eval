"""Generic scoring functions for eval tasks.

These are not NMDC-specific — they work for any eval that compares
predicted vs expected sets.
"""

from __future__ import annotations

from typing import Any


def score_sets(
    predicted: set[str],
    expected: set[str],
    exclude_from_precision: set[str] | None = None,
) -> dict[str, Any]:
    """Precision, recall, F1 on two sets of strings.

    Args:
        predicted: what the model returned
        expected: the ground truth
        exclude_from_precision: items to remove from predicted before computing
            precision (but NOT recall). Use for items that are correct but not
            in the ground truth by design (e.g. always-recommended fields).

    Returns dict with precision, recall, f1, true_positives, false_positives,
    false_negatives, and excluded_correct.
    """
    if not predicted and not expected:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not expected:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    precision_set = predicted - (exclude_from_precision or set())
    excluded_correct = predicted & (exclude_from_precision or set())

    tp = len(precision_set & expected)
    fp = len(precision_set - expected)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / len(expected) if expected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "true_positives": sorted(precision_set & expected),
        "false_positives": sorted(precision_set - expected),
        "false_negatives": sorted(expected - predicted),
        "excluded_correct": sorted(excluded_correct),
    }

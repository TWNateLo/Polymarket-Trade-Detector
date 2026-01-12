"""Evaluation utilities for insider detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from .ensemble import EnsembleScore


@dataclass(frozen=True)
class EvaluationResult:
    precision: float
    recall: float
    f1: float


def compute_classification_metrics(
    predictions: Sequence[EnsembleScore],
    ground_truth: Mapping[str, int],
    threshold: float = 0.5,
) -> EvaluationResult:
    tp = fp = fn = 0
    for prediction in predictions:
        label = ground_truth.get(prediction.entity_id, 0)
        flagged = prediction.score >= threshold
        if flagged and label == 1:
            tp += 1
        elif flagged and label == 0:
            fp += 1
        elif not flagged and label == 1:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return EvaluationResult(precision=precision, recall=recall, f1=f1)

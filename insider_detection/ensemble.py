"""Model ensembling utilities."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from statistics import mean
from typing import Dict, Iterable, List, Mapping

from .model_zoo import ModelPrediction


@dataclass(frozen=True)
class EnsembleScore:
    """Combined score for an entity after ensembling."""

    entity_id: str
    score: float
    breakdown: Mapping[str, float]


class EnsembleCombiner:
    """Combine predictions via weighted averages."""

    def __init__(self, weights: Mapping[str, float] | None = None) -> None:
        self.weights = dict(weights or {})

    def combine(self, predictions: Iterable[ModelPrediction]) -> List[EnsembleScore]:
        grouped: Dict[str, Dict[str, float]] = defaultdict(dict)
        for prediction in predictions:
            grouped[prediction.entity_id][prediction.model_name] = prediction.score

        results: List[EnsembleScore] = []
        for entity_id, model_scores in grouped.items():
            combined_score = self._weighted_average(model_scores)
            results.append(
                EnsembleScore(
                    entity_id=entity_id,
                    score=combined_score,
                    breakdown=model_scores,
                )
            )
        return results

    def aggregate_statistics(
        self, predictions: Iterable[ModelPrediction]
    ) -> Mapping[str, float]:
        by_model: Dict[str, List[float]] = defaultdict(list)
        for prediction in predictions:
            by_model[prediction.model_name].append(prediction.score)
        return {name: mean(scores) for name, scores in by_model.items() if scores}

    def _weighted_average(self, model_scores: Mapping[str, float]) -> float:
        if not model_scores:
            return 0.0

        if not self.weights:
            return mean(model_scores.values())

        total_weight = 0.0
        weighted_sum = 0.0
        for name, score in model_scores.items():
            weight = self.weights.get(name, 0.0)
            total_weight += weight
            weighted_sum += weight * score
        return weighted_sum / total_weight if total_weight else mean(model_scores.values())

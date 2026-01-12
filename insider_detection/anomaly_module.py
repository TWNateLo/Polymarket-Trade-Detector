"""Anomaly detection abstractions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Mapping, Protocol

from .feature_store import FeatureVector


class AnomalyDetector(Protocol):
    """Protocol implemented by anomaly detection algorithms."""

    name: str

    def score(self, features: Mapping[str, float]) -> float:
        ...


@dataclass(frozen=True)
class AnomalyScore:
    entity_id: str
    detector_name: str
    score: float


class AnomalyModule:
    """Coordinates multiple anomaly detectors."""

    def __init__(self, detectors: Iterable[AnomalyDetector]) -> None:
        self.detectors = list(detectors)

    def run(self, features: Iterable[FeatureVector]) -> List[AnomalyScore]:
        scores: List[AnomalyScore] = []
        for vector in features:
            for detector in self.detectors:
                score = detector.score(vector.features)
                scores.append(
                    AnomalyScore(
                        entity_id=vector.entity_id,
                        detector_name=detector.name,
                        score=score,
                    )
                )
        return scores

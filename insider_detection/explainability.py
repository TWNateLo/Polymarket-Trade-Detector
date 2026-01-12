"""Explainability utilities for insider detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Mapping

from .ensemble import EnsembleScore


@dataclass(frozen=True)
class Explanation:
    entity_id: str
    top_features: List[str]
    narrative: str


class ExplainabilityModule:
    """Produces lightweight explanations based on feature contributions."""

    def build_explanations(
        self, features: Mapping[str, float], ensemble_score: EnsembleScore
    ) -> Explanation:
        sorted_features = sorted(
            features.items(), key=lambda item: abs(item[1]), reverse=True
        )
        top_features = [name for name, _ in sorted_features[:3]]
        narrative = (
            f"Ensemble score {ensemble_score.score:.2f} derived from models: "
            + ", ".join(
                f"{model}={score:.2f}" for model, score in ensemble_score.breakdown.items()
            )
        )
        return Explanation(
            entity_id=ensemble_score.entity_id,
            top_features=top_features,
            narrative=narrative,
        )

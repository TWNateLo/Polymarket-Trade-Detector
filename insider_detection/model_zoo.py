"""Model management for the insider detection system."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Iterator, Mapping, Protocol

from .feature_store import FeatureVector


class PredictiveModel(Protocol):
    """Protocol for prediction models."""

    name: str

    def predict_proba(self, features: Mapping[str, float]) -> float:
        ...


@dataclass(frozen=True)
class ModelPrediction:
    """Capture an individual model's output."""

    model_name: str
    entity_id: str
    score: float
    metadata: Mapping[str, float]


@dataclass
class ModelWrapper:
    """Wraps a predictive model with standardized inference."""

    name: str
    model: PredictiveModel
    postprocess: Callable[[float], float] | None = None

    def predict(self, feature_vector: FeatureVector) -> ModelPrediction:
        raw_score = self.model.predict_proba(feature_vector.features)
        score = self.postprocess(raw_score) if self.postprocess else raw_score
        return ModelPrediction(
            model_name=self.name,
            entity_id=feature_vector.entity_id,
            score=score,
            metadata={"raw_score": raw_score},
        )


class ModelZoo:
    """Registry and iterator for heterogeneous models."""

    def __init__(self, models: Iterable[ModelWrapper] | None = None) -> None:
        self._models: Dict[str, ModelWrapper] = {}
        if models:
            for model in models:
                self.register(model)

    def register(self, wrapper: ModelWrapper) -> None:
        if wrapper.name in self._models:
            raise ValueError(f"Model {wrapper.name} already registered")
        self._models[wrapper.name] = wrapper

    def iter_models(self) -> Iterator[ModelWrapper]:
        return iter(self._models.values())

    def get(self, name: str) -> ModelWrapper:
        return self._models[name]

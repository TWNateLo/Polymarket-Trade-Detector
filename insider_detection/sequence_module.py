"""Temporal modeling components for insider detection."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List

from .data_ingestion import TradeEvent
from .feature_store import FeatureVector


@dataclass
class SequenceEmbedding:
    """Embedding representing temporal behavior."""

    entity_id: str
    values: List[float]


class SequenceModule:
    """Generates temporal embeddings via simplified transformers."""

    def __init__(self, embedding_dim: int = 8) -> None:
        self.embedding_dim = embedding_dim

    def encode(self, trades: Iterable[TradeEvent]) -> List[SequenceEmbedding]:
        sequences: dict[str, List[TradeEvent]] = {}
        for trade in trades:
            sequences.setdefault(trade.account_id, []).append(trade)

        embeddings: List[SequenceEmbedding] = []
        for entity_id, trade_seq in sequences.items():
            embedding = self._positional_encoding(trade_seq)
            embeddings.append(SequenceEmbedding(entity_id, embedding))
        return embeddings

    def enrich_features(
        self, features: Iterable[FeatureVector], embeddings: Iterable[SequenceEmbedding]
    ) -> List[FeatureVector]:
        embed_lookup = {embedding.entity_id: embedding for embedding in embeddings}
        enriched: List[FeatureVector] = []
        for vector in features:
            embedding = embed_lookup.get(vector.entity_id)
            if embedding:
                augmented = dict(vector.features)
                for idx, value in enumerate(embedding.values):
                    augmented[f"seq_{idx}"] = value
                enriched.append(FeatureVector(vector.entity_id, augmented, vector.as_of))
            else:
                enriched.append(vector)
        return enriched

    def _positional_encoding(self, trades: List[TradeEvent]) -> List[float]:
        values = [0.0] * self.embedding_dim
        for position, trade in enumerate(sorted(trades, key=lambda t: t.timestamp)):
            for idx in range(self.embedding_dim):
                angle = position / (10000 ** (2 * (idx // 2) / self.embedding_dim))
                if idx % 2 == 0:
                    values[idx] += math.sin(angle)
                else:
                    values[idx] += math.cos(angle)
        return values

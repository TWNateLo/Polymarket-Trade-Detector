"""Feature computation and storage for insider detection."""

from __future__ import annotations

import datetime as dt
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, List, Mapping, MutableMapping

from .data_ingestion import TradeEvent


@dataclass
class FeatureVector:
    """Encapsulates computed features for a trade or account."""

    entity_id: str
    features: Mapping[str, float]
    as_of: dt.datetime


@dataclass
class FeatureStore:
    """In-memory feature store suitable for experiments."""

    storage: MutableMapping[str, List[FeatureVector]] = field(
        default_factory=lambda: defaultdict(list)
    )

    def compute_features(self, trades: Iterable[TradeEvent]) -> List[FeatureVector]:
        """Generate rolling features based on trade history."""
        computed: List[FeatureVector] = []
        for trade in trades:
            features = {
                "avg_trade_size": self._rolling_average(trade.account_id, trade.size),
                "profit_proxy": self._profit_proxy(trade),
                "time_to_resolution_est": self._time_to_resolution_proxy(trade),
            }
            vector = FeatureVector(trade.account_id, features, trade.timestamp)
            self.storage[trade.account_id].append(vector)
            computed.append(vector)
        return computed

    def latest_features(self, entity_id: str) -> FeatureVector | None:
        """Retrieve the most recent feature vector for an entity."""
        vectors = self.storage.get(entity_id)
        return vectors[-1] if vectors else None

    def _rolling_average(self, account_id: str, new_value: float) -> float:
        history = self.storage.get(account_id, [])
        if not history:
            return new_value
        last = history[-1].features.get("avg_trade_size", new_value)
        return 0.5 * last + 0.5 * new_value

    def _profit_proxy(self, trade: TradeEvent) -> float:
        """Approximate profitability when final outcomes are unknown."""
        direction = 1.0 if trade.outcome.lower() in {"yes", "win"} else -1.0
        return direction * (1.0 - trade.price)

    def _time_to_resolution_proxy(self, trade: TradeEvent) -> float:
        """Placeholder estimate of time remaining to resolution."""
        midnight = dt.datetime.combine(trade.timestamp.date(), dt.time.min)
        delta = trade.timestamp - midnight
        return max(delta.total_seconds(), 1.0)

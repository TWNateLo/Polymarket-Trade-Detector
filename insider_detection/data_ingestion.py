"""Data ingestion utilities for Polymarket insider detection."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Protocol


@dataclass(frozen=True)
class TradeEvent:
    """Represents a single trade on Polymarket."""

    trade_id: str
    account_id: str
    market_id: str
    timestamp: dt.datetime
    outcome: str
    size: float
    price: float


@dataclass(frozen=True)
class MarketResolution:
    """Represents a market resolution event."""

    market_id: str
    resolution_time: dt.datetime
    resolved_outcome: str


class TradeStreamSource(Protocol):
    """Protocol for streaming trade events from an external system."""

    def stream_trades(self) -> Iterator[TradeEvent]:
        ...


class ResolutionStreamSource(Protocol):
    """Protocol for streaming market resolutions."""

    def stream_resolutions(self) -> Iterator[MarketResolution]:
        ...


class DataIngestion:
    """Coordinates ingestion across multiple sources and adapters."""

    def __init__(
        self,
        trade_sources: Iterable[TradeStreamSource],
        resolution_sources: Iterable[ResolutionStreamSource],
    ) -> None:
        self._trade_sources = list(trade_sources)
        self._resolution_sources = list(resolution_sources)

    def get_recent_trades(self) -> Iterator[TradeEvent]:
        """Yield recent trade events from all sources."""
        for source in self._trade_sources:
            yield from source.stream_trades()

    def get_recent_resolutions(self) -> Iterator[MarketResolution]:
        """Yield recent market resolutions from all sources."""
        for source in self._resolution_sources:
            yield from source.stream_resolutions()

    def snapshot_trades(self) -> List[TradeEvent]:
        """Materialize a list of trades for batch pipelines."""
        return list(self.get_recent_trades())

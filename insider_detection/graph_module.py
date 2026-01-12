"""Graph analytics for identifying coordinated wallets."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Set

from .data_ingestion import TradeEvent


@dataclass(frozen=True)
class GraphEdge:
    source: str
    target: str
    weight: float


class GraphModule:
    """Builds and analyzes wallet relationship graphs."""

    def __init__(self, threshold: float = 0.7) -> None:
        self.threshold = threshold

    def build_wallet_graph(self, trades: Iterable[TradeEvent]) -> Mapping[str, List[GraphEdge]]:
        adjacency: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        trades_by_market: Dict[str, List[TradeEvent]] = defaultdict(list)
        for trade in trades:
            trades_by_market[trade.market_id].append(trade)

        for market_id, market_trades in trades_by_market.items():
            for i, trade_i in enumerate(market_trades):
                for trade_j in market_trades[i + 1 :]:
                    weight = self._co_trading_weight(trade_i, trade_j)
                    if weight >= self.threshold:
                        adjacency[trade_i.account_id][trade_j.account_id] += weight
                        adjacency[trade_j.account_id][trade_i.account_id] += weight
        return {
            node: [GraphEdge(node, target, weight) for target, weight in edges.items()]
            for node, edges in adjacency.items()
        }

    def detect_communities(
        self, graph: Mapping[str, Iterable[GraphEdge]]
    ) -> List[Set[str]]:
        """Detect connected components as proxy communities."""
        visited: Set[str] = set()
        communities: List[Set[str]] = []

        for node in graph:
            if node in visited:
                continue
            community = set()
            queue: deque[str] = deque([node])
            visited.add(node)
            while queue:
                current = queue.popleft()
                community.add(current)
                for edge in graph.get(current, []):
                    if edge.target not in visited:
                        visited.add(edge.target)
                        queue.append(edge.target)
            communities.append(community)
        return communities

    def _co_trading_weight(self, a: TradeEvent, b: TradeEvent) -> float:
        """Estimate coordination weight based on trade proximity."""
        same_direction = 1.0 if a.outcome == b.outcome else 0.5
        size_similarity = 1.0 - abs(a.size - b.size) / max(a.size, b.size, 1.0)
        time_gap = abs((a.timestamp - b.timestamp).total_seconds()) + 1.0
        return same_direction * size_similarity / time_gap

"""Pipeline orchestration for the insider detection system."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Mapping, Optional

from .data_ingestion import DataIngestion, TradeEvent
from .feature_store import FeatureStore, FeatureVector
from .model_zoo import ModelPrediction, ModelZoo
from .ensemble import EnsembleCombiner, EnsembleScore
from .interfaces.alerts import AlertDispatcher, Alert
from .sequence_module import SequenceModule
from .graph_module import GraphModule
from .anomaly_module import AnomalyModule, AnomalyScore
from .explainability import ExplainabilityModule, Explanation


@dataclass
class InsiderDetectionPipeline:
    """Coordinates end-to-end inference, scoring, and alerting."""

    data_ingestion: DataIngestion
    feature_store: FeatureStore
    model_zoo: ModelZoo
    ensemble: EnsembleCombiner
    alert_dispatcher: AlertDispatcher
    sequence_module: SequenceModule | None = None
    graph_module: GraphModule | None = None
    anomaly_module: AnomalyModule | None = None
    explainability: ExplainabilityModule | None = None
    markets_of_interest: Optional[Iterable[str]] = None
    alerts: List[Alert] = field(default_factory=list, init=False)
    explanations: List[Explanation] = field(default_factory=list, init=False)
    anomaly_scores: List[AnomalyScore] = field(default_factory=list, init=False)

    def run_inference(self) -> List[Alert]:
        """Execute inference across configured models and produce alerts."""
        trade_events = self._load_trade_events()
        feature_vectors = self.feature_store.compute_features(trade_events)
        feature_vectors = self._enrich_with_sequences(trade_events, feature_vectors)

        predictions = self._generate_predictions(feature_vectors)
        self.anomaly_scores = self._run_anomaly_detectors(feature_vectors)

        combined_scores = self.ensemble.combine(predictions)
        self.explanations = self._build_explanations(feature_vectors, combined_scores)

        self.alerts = self.alert_dispatcher.create_alerts(combined_scores)
        self.alert_dispatcher.dispatch(self.alerts)
        return self.alerts

    def _load_trade_events(self) -> List[TradeEvent]:
        """Load trade events scoped by markets of interest."""
        if self.markets_of_interest is None:
            return list(self.data_ingestion.get_recent_trades())

        market_set = set(self.markets_of_interest)
        return [
            trade
            for trade in self.data_ingestion.get_recent_trades()
            if trade.market_id in market_set
        ]

    def _enrich_with_sequences(
        self, trades: Iterable[TradeEvent], features: List[FeatureVector]
    ) -> List[FeatureVector]:
        if self.sequence_module is None:
            return features
        embeddings = self.sequence_module.encode(trades)
        return self.sequence_module.enrich_features(features, embeddings)

    def _generate_predictions(
        self, feature_vectors: Iterable[FeatureVector]
    ) -> List[ModelPrediction]:
        predictions: List[ModelPrediction] = []
        for features in feature_vectors:
            for model in self.model_zoo.iter_models():
                prediction = model.predict(features)
                predictions.append(prediction)
        return predictions

    def _run_anomaly_detectors(
        self, feature_vectors: Iterable[FeatureVector]
    ) -> List[AnomalyScore]:
        if self.anomaly_module is None:
            return []
        return self.anomaly_module.run(feature_vectors)

    def _build_explanations(
        self, feature_vectors: Iterable[FeatureVector], scores: Iterable[EnsembleScore]
    ) -> List[Explanation]:
        if self.explainability is None:
            return []
        feature_lookup = {vector.entity_id: vector for vector in feature_vectors}
        explanations: List[Explanation] = []
        for score in scores:
            vector = feature_lookup.get(score.entity_id)
            if vector is None:
                continue
            explanations.append(
                self.explainability.build_explanations(vector.features, score)
            )
        return explanations

    def run_backtest(self, historical_trades: Iterable[TradeEvent]) -> Mapping[str, float]:
        """Run the ensemble on historical data to compute diagnostics."""
        trades_list = list(historical_trades)
        feature_vectors = self.feature_store.compute_features(trades_list)
        feature_vectors = self._enrich_with_sequences(trades_list, feature_vectors)

        predictions = self._generate_predictions(feature_vectors)
        metrics = dict(self.ensemble.aggregate_statistics(predictions))

        if self.graph_module is not None:
            graph = self.graph_module.build_wallet_graph(trades_list)
            communities = self.graph_module.detect_communities(graph)
            metrics["communities_detected"] = float(len(communities))

        anomaly_scores = self._run_anomaly_detectors(feature_vectors)
        if anomaly_scores:
            metrics["avg_anomaly_score"] = sum(
                score.score for score in anomaly_scores
            ) / len(anomaly_scores)

        return metrics

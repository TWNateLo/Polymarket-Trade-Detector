"""Alerting interfaces for the insider detection system."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from ..ensemble import EnsembleScore


@dataclass(frozen=True)
class Alert:
    entity_id: str
    score: float
    severity: str
    message: str


class AlertDispatcher:
    """Dispatch alerts to downstream systems."""

    def __init__(self, critical_threshold: float = 0.9, high_threshold: float = 0.7) -> None:
        self.critical_threshold = critical_threshold
        self.high_threshold = high_threshold
        self.sent_alerts: List[Alert] = []

    def create_alerts(self, ensemble_scores: Iterable[EnsembleScore]) -> List[Alert]:
        alerts: List[Alert] = []
        for score in ensemble_scores:
            severity = self._determine_severity(score.score)
            if severity == "info":
                continue
            message = (
                f"Account {score.entity_id} flagged with severity {severity} "
                f"(score={score.score:.2f})."
            )
            alerts.append(
                Alert(
                    entity_id=score.entity_id,
                    score=score.score,
                    severity=severity,
                    message=message,
                )
            )
        return alerts

    def dispatch(self, alerts: Iterable[Alert]) -> None:
        self.sent_alerts.extend(alerts)

    def _determine_severity(self, score: float) -> str:
        if score >= self.critical_threshold:
            return "critical"
        if score >= self.high_threshold:
            return "high"
        if score >= 0.5:
            return "medium"
        return "info"

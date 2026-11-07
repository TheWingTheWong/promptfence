"""Layer 4: combine evidence from all layers into one risk score.

Each distinct rule contributes its severity weight once (repeating the same
phrase ten times is not ten times more dangerous). Rule evidence and the
classifier probability are combined with a noisy-OR::

    risk = 1 - (1 - w_1)(1 - w_2)...(1 - w_k) * (1 - c * p)

where ``w_i`` are per-rule weights, ``p`` is the classifier probability and
``c`` caps how much the classifier alone can contribute. Independent weak
signals therefore accumulate, while no single weak signal saturates the score.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .types import Finding, Layer, Verdict


@dataclass(frozen=True)
class ScoringConfig:
    # Defaults were chosen by 5-fold cross-validation on the *train* split
    # (scripts/eval.py reports the untouched test split).
    #: Maximum contribution of the classifier on its own (``c`` above).
    classifier_weight: float = 0.7
    #: Risk at or above which the verdict is at least "suspicious".
    suspicious_threshold: float = 0.4
    #: Risk at or above which the verdict is "malicious".
    malicious_threshold: float = 0.6
    #: Sentence-level classifier probability at which a passage is highlighted.
    classifier_span_threshold: float = 0.8

    def __post_init__(self) -> None:
        if not 0.0 <= self.classifier_weight <= 1.0:
            raise ValueError("classifier_weight must be in [0, 1]")
        if not 0.0 < self.suspicious_threshold <= self.malicious_threshold <= 1.0:
            raise ValueError("need 0 < suspicious_threshold <= malicious_threshold <= 1")


def combine(
    findings: Iterable[Finding],
    classifier_score: float | None,
    config: ScoringConfig,
) -> float:
    best: dict[str, float] = {}
    for f in findings:
        if f.layer is Layer.CLASSIFIER:
            continue
        best[f.rule_id] = max(best.get(f.rule_id, 0.0), f.weight)
    keep = 1.0
    for weight in best.values():
        keep *= 1.0 - weight
    if classifier_score is not None:
        keep *= 1.0 - config.classifier_weight * classifier_score
    return 1.0 - keep


def verdict_for(risk: float, config: ScoringConfig) -> Verdict:
    if risk >= config.malicious_threshold:
        return Verdict.MALICIOUS
    if risk >= config.suspicious_threshold:
        return Verdict.SUSPICIOUS
    return Verdict.BENIGN

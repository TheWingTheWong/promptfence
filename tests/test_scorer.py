import pytest

from promptfence.scorer import ScoringConfig, combine, verdict_for
from promptfence.types import Finding, Layer, Severity, Verdict, merge_spans


def finding(rule_id, severity, layer=Layer.RULES):
    return Finding(rule_id, "n", "c", severity, "e", 0, 1, "x", layer, (), severity.weight)


def test_noisy_or_combination():
    cfg = ScoringConfig(classifier_weight=0.5)
    risk = combine([finding("A", Severity.HIGH), finding("B", Severity.MEDIUM)], 0.4, cfg)
    assert risk == pytest.approx(1 - (1 - 0.6) * (1 - 0.35) * (1 - 0.2))


def test_same_rule_counts_once_and_classifier_findings_ignored():
    cfg = ScoringConfig()
    fs = [finding("A", Severity.LOW), finding("A", Severity.HIGH),
          finding("ML", Severity.HIGH, Layer.CLASSIFIER)]  # fmt: skip
    assert combine(fs, None, cfg) == pytest.approx(0.6)


def test_verdict_thresholds():
    cfg = ScoringConfig(suspicious_threshold=0.3, malicious_threshold=0.6)
    assert verdict_for(0.1, cfg) is Verdict.BENIGN
    assert verdict_for(0.3, cfg) is Verdict.SUSPICIOUS
    assert verdict_for(0.6, cfg) is Verdict.MALICIOUS


@pytest.mark.parametrize(
    "kwargs",
    [{"classifier_weight": 1.5}, {"suspicious_threshold": 0.7, "malicious_threshold": 0.5},
     {"suspicious_threshold": 0.0}],
)  # fmt: skip
def test_config_validation(kwargs):
    with pytest.raises(ValueError):
        ScoringConfig(**kwargs)


def test_severity_helpers():
    assert Severity.parse("High") is Severity.HIGH
    assert Severity.CRITICAL.escalate() is Severity.CRITICAL
    with pytest.raises(ValueError):
        Severity.parse("severe")


def test_merge_spans():
    assert merge_spans([(5, 8), (0, 2), (1, 3), (8, 9)]) == [(0, 3), (5, 9)]

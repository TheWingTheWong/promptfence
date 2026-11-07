"""Core data types shared by every detection layer."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any


class Severity(IntEnum):
    """How dangerous a single finding is on its own.

    Each level carries a *weight* in ``[0, 1]`` that the scorer treats as the
    probability that this finding alone indicates an attack.
    """

    INFO = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

    @property
    def weight(self) -> float:
        return _SEVERITY_WEIGHTS[self]

    @property
    def label(self) -> str:
        return self.name.lower()

    def escalate(self, steps: int = 1) -> Severity:
        return Severity(min(int(self) + steps, int(Severity.CRITICAL)))

    @classmethod
    def parse(cls, value: str | int | Severity) -> Severity:
        if isinstance(value, Severity):
            return value
        if isinstance(value, int):
            return cls(value)
        try:
            return cls[value.strip().upper()]
        except KeyError:
            valid = ", ".join(s.label for s in cls)
            raise ValueError(f"unknown severity {value!r} (expected one of: {valid})") from None


_SEVERITY_WEIGHTS = {
    Severity.INFO: 0.03,
    Severity.LOW: 0.12,
    Severity.MEDIUM: 0.35,
    Severity.HIGH: 0.6,
    Severity.CRITICAL: 0.85,
}


class Verdict(str, Enum):
    BENIGN = "benign"
    SUSPICIOUS = "suspicious"
    MALICIOUS = "malicious"

    def __str__(self) -> str:
        return self.value


class Layer(str, Enum):
    """Which detection layer produced a finding."""

    NORMALIZER = "normalizer"
    RULES = "rules"
    CLASSIFIER = "classifier"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class Finding:
    """A single piece of evidence, anchored to a span of the *original* text."""

    rule_id: str
    name: str
    category: str
    severity: Severity
    explanation: str
    start: int
    end: int
    excerpt: str
    layer: Layer = Layer.RULES
    #: Obfuscation layers the match was found through, outermost first,
    #: e.g. ``("html-comment",)`` or ``("base64",)``.
    via: tuple[str, ...] = ()
    #: Contribution to the overall risk. Classifier findings are informational
    #: (the classifier contributes through ``Report.classifier_score``).
    weight: float = 0.0

    @property
    def span(self) -> tuple[int, int]:
        return (self.start, self.end)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "category": self.category,
            "severity": self.severity.label,
            "explanation": self.explanation,
            "start": self.start,
            "end": self.end,
            "excerpt": self.excerpt,
            "layer": self.layer.value,
            "via": list(self.via),
            "weight": round(self.weight, 4),
        }


@dataclass
class Report:
    """The result of :func:`promptfence.scan`."""

    text: str
    risk: float
    verdict: Verdict
    findings: list[Finding] = field(default_factory=list)
    classifier_score: float | None = None
    signals: dict[str, int] = field(default_factory=dict)
    elapsed_ms: float = 0.0

    @property
    def is_malicious(self) -> bool:
        return self.verdict is Verdict.MALICIOUS

    @property
    def is_suspicious(self) -> bool:
        """True for anything that is not clearly benign."""
        return self.verdict is not Verdict.BENIGN

    @property
    def max_severity(self) -> Severity | None:
        scored = [f.severity for f in self.findings if f.layer is not Layer.CLASSIFIER]
        return max(scored) if scored else None

    def spans(
        self,
        min_severity: Severity = Severity.LOW,
        *,
        include_classifier: bool = False,
    ) -> list[tuple[int, int]]:
        """Merged, sorted spans of the original text that carry findings."""
        selected = (
            f
            for f in self.findings
            if f.severity >= min_severity
            and (include_classifier or f.layer is not Layer.CLASSIFIER)
        )
        return merge_spans(f.span for f in selected)

    def highlight(
        self,
        open_mark: str = "[[",
        close_mark: str = "]]",
        min_severity: Severity = Severity.LOW,
    ) -> str:
        """Return the original text with suspicious spans wrapped in markers."""
        out: list[str] = []
        cursor = 0
        for start, end in self.spans(min_severity):
            out.append(self.text[cursor:start])
            out.append(open_mark + self.text[start:end] + close_mark)
            cursor = end
        out.append(self.text[cursor:])
        return "".join(out)

    def to_dict(self) -> dict[str, Any]:
        return {
            "risk": round(self.risk, 4),
            "verdict": self.verdict.value,
            "classifier_score": (
                None if self.classifier_score is None else round(self.classifier_score, 4)
            ),
            "signals": dict(self.signals),
            "findings": [f.to_dict() for f in self.findings],
            "spans": [list(s) for s in self.spans()],
            "elapsed_ms": round(self.elapsed_ms, 3),
        }


def merge_spans(spans: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping or touching ``(start, end)`` spans."""
    merged: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            prev_start, prev_end = merged[-1]
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged

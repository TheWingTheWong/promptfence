"""Orchestrates the four layers: normalize -> rules -> classifier -> score."""

from __future__ import annotations

import time
from functools import lru_cache

from .classifier import NgramClassifier
from .normalize import ENCODED_KINDS, HIDDEN_KINDS, Artifact, Normalized, View, normalize
from .rules import RuleSet
from .scorer import ScoringConfig, combine, verdict_for
from .types import Finding, Layer, Report, Severity

#: Character-level tricks that, when they overlap a rule match, escalate it.
_EVASION_KINDS = frozenset({"homoglyph", "invisible", "spaced-letters", "leetspeak"})
_EXCERPT_CHARS = 160
_LONG_VIEW_CHARS = 400


def classifier_text(norm: Normalized) -> str:
    """The text the classifier sees: every normalized view except raw markup."""
    return "\n".join(v.text for v in norm.views if v.name != "markup")


class Detector:
    """A configured scanner. Reuse one instance; construction loads the model.

    Args:
        rules: Rule set to apply. Defaults to the bundled pack.
        classifier: A trained classifier, ``None`` for the bundled model, or
            ``False`` to run rules only.
        config: Scoring weights and thresholds.
        max_chars: Inputs longer than this are truncated before scanning
            (a ``truncated`` signal is recorded) to bound latency.
    """

    def __init__(
        self,
        rules: RuleSet | None = None,
        classifier: NgramClassifier | bool | None = None,
        config: ScoringConfig | None = None,
        *,
        max_chars: int = 200_000,
    ) -> None:
        self.rules = rules if rules is not None else RuleSet.default()
        if classifier is None or classifier is True:
            self.classifier: NgramClassifier | None = NgramClassifier.load()
        elif classifier is False:
            self.classifier = None
        else:
            self.classifier = classifier
        self.config = config or ScoringConfig()
        self.max_chars = max_chars

    # ------------------------------------------------------------------ API
    def scan(self, text: str) -> Report:
        if not isinstance(text, str):
            raise TypeError(f"scan() expects str, got {type(text).__name__}")
        started = time.perf_counter()
        scanned = text[: self.max_chars]
        norm = normalize(scanned)

        findings = _dedupe(self._rule_findings(norm) + self._normalizer_findings(norm))
        score, ml_findings = self._classify(norm)
        findings = sorted(findings + ml_findings, key=lambda f: (-f.severity, f.start, f.rule_id))

        risk = combine(findings, score, self.config)
        signals = norm.signals
        if len(text) > self.max_chars:
            signals["truncated"] = len(text) - self.max_chars
        return Report(
            text=text,
            risk=risk,
            verdict=verdict_for(risk, self.config),
            findings=findings,
            classifier_score=score,
            signals=signals,
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )

    # --------------------------------------------------------------- layers
    def _rule_findings(self, norm: Normalized) -> list[Finding]:
        findings = []
        for view in norm.views:
            for match in self.rules.match(view.text):
                start, end = view.mapped.span(match.start, match.end)
                via = list(view.via)
                hidden = norm.hidden_kind(start, end)
                if hidden:
                    via.insert(0, hidden)
                for art in _overlapping(norm.artifacts, start, end, _EVASION_KINDS):
                    if art.kind not in via:
                        via.append(art.kind)

                severity = match.rule.severity.escalate() if via else match.rule.severity
                matched = view.text[match.start : match.end]
                explanation = match.rule.explanation
                if via:
                    explanation += (
                        f" Found via {', '.join(via)}, which suggests deliberate concealment."
                    )
                findings.append(
                    Finding(
                        rule_id=match.rule.id,
                        name=match.rule.name,
                        category=match.rule.category,
                        severity=severity,
                        explanation=explanation,
                        start=start,
                        end=end,
                        excerpt=_excerpt(
                            matched if _is_encoded(view) else norm.original[start:end]
                        ),
                        layer=Layer.RULES,
                        via=tuple(via),
                        weight=severity.weight,
                    )
                )
        return findings

    def _normalizer_findings(self, norm: Normalized) -> list[Finding]:
        findings = []
        homoglyphs = [a for a in norm.artifacts if a.kind == "homoglyph"]
        invisible = [a for a in norm.artifacts if a.kind == "invisible"]

        def add(art: Artifact, rule_id: str, name: str, severity: Severity, why: str) -> None:
            excerpt = art.detail or norm.original[art.start : art.end]
            findings.append(
                Finding(
                    rule_id=rule_id,
                    name=name,
                    category="obfuscation",
                    severity=severity,
                    explanation=why,
                    start=art.start,
                    end=art.end,
                    excerpt=_excerpt(excerpt),
                    layer=Layer.NORMALIZER,
                    via=(art.kind,),
                    weight=severity.weight,
                )
            )

        for art in norm.artifacts:
            if art.kind == "unicode-tags":
                add(art, "PF-NRM-001", "unicode-tag-smuggling", Severity.HIGH,
                    "Invisible Unicode tag characters encode hidden ASCII text. They render as "
                    "nothing but many models read them.")  # fmt: skip
            elif art.kind == "homoglyph":
                severity = Severity.MEDIUM if len(homoglyphs) >= 2 else Severity.LOW
                add(art, "PF-NRM-002", "mixed-script-homoglyphs", severity,
                    "A word mixes Latin with lookalike letters from another script, a common "
                    "way to defeat keyword filters.")  # fmt: skip
            elif art.kind == "invisible" and len(invisible) >= 3:
                add(
                    art,
                    "PF-NRM-003",
                    "invisible-characters",
                    Severity.LOW,
                    "Zero-width or bidirectional control characters are embedded inside words.",
                )
            elif art.kind in ("base64", "hex", "rot13", "unicode-escape"):
                add(art, "PF-NRM-004", "encoded-natural-language", Severity.LOW,
                    f"A {art.kind} blob decodes to readable text. Legitimate content rarely "
                    "encodes prose.")  # fmt: skip
            elif art.kind in HIDDEN_KINDS - {"html-attribute"} and _word_count(norm, art) >= 4:
                add(art, "PF-NRM-005", "hidden-text", Severity.INFO,
                    f"Text inside {art.kind} is not shown to human readers but is visible to "
                    "a model.")  # fmt: skip
        return findings

    def _classify(self, norm: Normalized) -> tuple[float | None, list[Finding]]:
        if self.classifier is None:
            return None, []
        score = self.classifier.predict_proba(classifier_text(norm))
        findings = []
        for view in norm.views:
            if view.name == "markup":
                continue
            # Sentence-level scores only feed the document score for long
            # inputs, where one injected sentence would otherwise be diluted.
            long_view = len(view.text) > _LONG_VIEW_CHARS
            for chunk in self.classifier.score_chunks(view.text):
                if long_view:
                    score = max(score, chunk.probability)
                if chunk.probability < self.config.classifier_span_threshold:
                    continue
                start, end = view.mapped.span(chunk.start, chunk.end)
                text = view.text[chunk.start : chunk.end].strip()
                findings.append(
                    Finding(
                        rule_id="PF-ML-001",
                        name="ngram-classifier",
                        category="classifier",
                        severity=_severity_for_probability(chunk.probability),
                        explanation=(
                            f"The n-gram classifier scored this passage "
                            f"{chunk.probability:.2f} for injection-like language."
                        ),
                        start=start,
                        end=end,
                        excerpt=_excerpt(text),
                        layer=Layer.CLASSIFIER,
                        via=view.via,
                        weight=0.0,
                    )
                )
        return score, _dedupe(findings)


def _severity_for_probability(p: float) -> Severity:
    if p >= 0.95:
        return Severity.HIGH
    if p >= 0.85:
        return Severity.MEDIUM
    return Severity.LOW


def _is_encoded(view: View) -> bool:
    return any(kind in ENCODED_KINDS for kind in view.via)


def _overlapping(artifacts: list[Artifact], start: int, end: int, kinds: frozenset[str]):
    return [a for a in artifacts if a.kind in kinds and a.start < end and start < a.end]


def _word_count(norm: Normalized, art: Artifact) -> int:
    return len(norm.original[art.start : art.end].split())


def _excerpt(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= _EXCERPT_CHARS else text[: _EXCERPT_CHARS - 1] + "…"


def _dedupe(findings: list[Finding]) -> list[Finding]:
    """Keep the most severe finding per rule among overlapping spans."""
    kept: list[Finding] = []
    for f in sorted(findings, key=lambda f: (-f.severity, -(f.end - f.start), f.start)):
        if any(k.rule_id == f.rule_id and k.start < f.end and f.start < k.end for k in kept):
            continue
        kept.append(f)
    return kept


@lru_cache(maxsize=1)
def default_detector() -> Detector:
    return Detector()


def scan(text: str) -> Report:
    """Scan ``text`` with the default rules, model and thresholds."""
    return default_detector().scan(text)

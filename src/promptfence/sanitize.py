"""Neutralize detected spans so the remaining text can be passed on.

Sanitizing is a mitigation, not a guarantee: it removes what was detected,
and the detector has false negatives. Prefer dropping a document flagged
``malicious`` altogether when your application can afford to.
"""

from __future__ import annotations

import re
from enum import Enum

from .detector import Detector, default_detector
from .normalize import HIDDEN_KINDS, normalize, strip_invisible
from .types import Finding, Layer, Report, Severity, merge_spans

_SENTENCE_END = re.compile(r"[.!?](?=\s)|\n")


class Mode(str, Enum):
    #: Replace each span with ``[promptfence: removed <rule ids>]``. Rule ids,
    #: not category names, so the placeholder itself reads as neutral text.
    REDACT = "redact"
    #: Delete each span.
    REMOVE = "remove"
    #: Keep the text but fence it so a downstream prompt can treat it as data.
    MARK = "mark"


class Scope(str, Enum):
    #: Only the characters a finding matched.
    MATCH = "match"
    #: Grow each span to the enclosing sentence, or the whole hidden region
    #: (HTML comment, hidden element) it sits in.
    SENTENCE = "sentence"


MARK_OPEN = "[untrusted-instruction]"
MARK_CLOSE = "[/untrusted-instruction]"


def sanitize(
    text: str,
    report: Report | None = None,
    *,
    mode: Mode | str = Mode.REDACT,
    scope: Scope | str = Scope.SENTENCE,
    min_severity: Severity | str = Severity.MEDIUM,
    include_classifier: bool = False,
    detector: Detector | None = None,
) -> str:
    """Return ``text`` with suspicious spans neutralized.

    Args:
        text: The untrusted input.
        report: A report for exactly this text; computed if omitted.
        mode: ``redact``, ``remove`` or ``mark``.
        scope: ``sentence`` (default) or ``match``.
        min_severity: Ignore findings below this severity.
        include_classifier: Also neutralize passages flagged only by the
            classifier (higher recall, more collateral damage).
        detector: Detector to use when ``report`` is omitted.

    Invisible characters and Unicode tag payloads are always stripped.
    """
    mode = Mode(mode)
    scope = Scope(scope)
    min_severity = Severity.parse(min_severity)
    if report is None:
        report = (detector or default_detector()).scan(text)
    elif report.text != text:
        raise ValueError("report was produced for a different text")

    selected = [
        f
        for f in report.findings
        if f.severity >= min_severity and (include_classifier or f.layer is not Layer.CLASSIFIER)
    ]
    spans = _spans(text, selected, scope)

    out: list[str] = []
    cursor = 0
    for start, end in spans:
        out.append(text[cursor:start])
        out.append(_replacement(text[start:end], _rule_ids(selected, start, end), mode))
        cursor = end
    out.append(text[cursor:])
    return strip_invisible("".join(out))


def _spans(text: str, findings: list[Finding], scope: Scope) -> list[tuple[int, int]]:
    raw = [f.span for f in findings]
    if scope is Scope.MATCH or not raw:
        return merge_spans(raw)
    hidden = [a for a in normalize(text).artifacts if a.kind in HIDDEN_KINDS]
    expanded = []
    for start, end in raw:
        region = next((a for a in hidden if a.start <= start and end <= a.end), None)
        if region is not None:
            expanded.append((region.start, region.end))
        else:
            expanded.append(_sentence_bounds(text, start, end))
    return merge_spans(expanded)


def _sentence_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    left = 0
    for m in _SENTENCE_END.finditer(text, 0, start):
        left = m.end()
    m = _SENTENCE_END.search(text, end)
    right = m.end() if m and m.group(0) != "\n" else (m.start() if m else len(text))
    while left < start and text[left].isspace():
        left += 1
    return (min(left, start), max(right, end))


def _rule_ids(findings: list[Finding], start: int, end: int) -> list[str]:
    return sorted({f.rule_id for f in findings if f.start < end and start < f.end})


def _replacement(original: str, rule_ids: list[str], mode: Mode) -> str:
    if mode is Mode.REMOVE:
        return ""
    if mode is Mode.MARK:
        return f"{MARK_OPEN}{original}{MARK_CLOSE}"
    return f"[promptfence: removed {', '.join(rule_ids)}]"

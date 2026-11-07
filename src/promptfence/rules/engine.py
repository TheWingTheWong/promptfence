"""Rule loading, validation and matching.

A rule pack is a YAML document::

    version: 1
    rules:
      - id: PF-OVR-001
        name: ignore-previous-instructions
        category: instruction_override
        severity: high
        explanation: Asks the model to discard the instructions it was given.
        patterns:
          - '\\b(?:ignore|disregard)\\s+(?:all\\s+)?previous\\s+instructions\\b'
        examples: ["Ignore all previous instructions."]
        counter_examples: ["Ignore the previous email, I sent it twice."]

Patterns are Python regular expressions compiled case-insensitively and in
multiline mode; use inline groups such as ``(?-i:...)`` to opt out. The
``examples`` and ``counter_examples`` are executable documentation: the test
suite asserts every example matches and no counter-example does.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from ..types import Severity

_ID = re.compile(r"^[A-Z][A-Z0-9]*(?:-[A-Z0-9]+)+$")
_FLAGS = re.IGNORECASE | re.MULTILINE


class RuleError(ValueError):
    """A rule pack is malformed."""


@dataclass(frozen=True)
class Rule:
    id: str
    name: str
    category: str
    severity: Severity
    explanation: str
    patterns: tuple[re.Pattern[str], ...]
    examples: tuple[str, ...] = ()
    counter_examples: tuple[str, ...] = ()

    def finditer(self, text: str) -> Iterator[tuple[int, int]]:
        for pattern in self.patterns:
            for m in pattern.finditer(text):
                if m.end() > m.start():
                    yield m.span()

    def matches(self, text: str) -> bool:
        return any(True for _ in self.finditer(text))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Rule:
        if not isinstance(data, dict):
            raise RuleError(f"rule must be a mapping, got {type(data).__name__}")
        rule_id = str(data.get("id", "")).strip()
        if not _ID.match(rule_id):
            raise RuleError(f"invalid rule id {rule_id!r} (expected e.g. 'PF-OVR-001')")
        missing = [k for k in ("name", "category", "severity", "explanation", "patterns")
                   if not data.get(k)]  # fmt: skip
        if missing:
            raise RuleError(f"{rule_id}: missing required field(s): {', '.join(missing)}")
        unknown = set(data) - {
            "id", "name", "category", "severity", "explanation", "patterns",
            "examples", "counter_examples",
        }  # fmt: skip
        if unknown:
            raise RuleError(f"{rule_id}: unknown field(s): {', '.join(sorted(unknown))}")
        try:
            severity = Severity.parse(data["severity"])
        except ValueError as exc:
            raise RuleError(f"{rule_id}: {exc}") from None

        raw_patterns = data["patterns"]
        if isinstance(raw_patterns, str):
            raw_patterns = [raw_patterns]
        compiled = []
        for raw in raw_patterns:
            try:
                compiled.append(re.compile(str(raw), _FLAGS))
            except re.error as exc:
                raise RuleError(f"{rule_id}: bad pattern {raw!r}: {exc}") from None

        return cls(
            id=rule_id,
            name=str(data["name"]),
            category=str(data["category"]),
            severity=severity,
            explanation=" ".join(str(data["explanation"]).split()),
            patterns=tuple(compiled),
            examples=tuple(str(x) for x in data.get("examples") or ()),
            counter_examples=tuple(str(x) for x in data.get("counter_examples") or ()),
        )


@dataclass(frozen=True)
class RuleMatch:
    rule: Rule
    start: int
    end: int


class RuleSet:
    """An ordered, id-unique collection of rules."""

    def __init__(self, rules: Iterable[Rule] = ()) -> None:
        self._rules: dict[str, Rule] = {}
        for rule in rules:
            if rule.id in self._rules:
                raise RuleError(f"duplicate rule id {rule.id}")
            self._rules[rule.id] = rule

    @classmethod
    def from_yaml(cls, source: str) -> RuleSet:
        try:
            doc = yaml.safe_load(source)
        except yaml.YAMLError as exc:
            raise RuleError(f"invalid YAML: {exc}") from None
        if not isinstance(doc, dict) or "rules" not in doc:
            raise RuleError("rule pack must be a mapping with a 'rules' list")
        if doc.get("version", 1) != 1:
            raise RuleError(f"unsupported rule pack version {doc.get('version')!r}")
        return cls(Rule.from_dict(r) for r in doc["rules"] or [])

    @classmethod
    def from_file(cls, path: str | Path) -> RuleSet:
        return cls.from_yaml(Path(path).read_text(encoding="utf-8"))

    @classmethod
    def default(cls) -> RuleSet:
        text = resources.files(__package__).joinpath("default.yaml").read_text("utf-8")
        return cls.from_yaml(text)

    def __iter__(self) -> Iterator[Rule]:
        return iter(self._rules.values())

    def __len__(self) -> int:
        return len(self._rules)

    def __contains__(self, rule_id: object) -> bool:
        return rule_id in self._rules

    def get(self, rule_id: str) -> Rule:
        return self._rules[rule_id]

    def merged(self, other: RuleSet) -> RuleSet:
        """Return a new set where rules in ``other`` override same-id rules here."""
        combined = dict(self._rules)
        combined.update({r.id: r for r in other})
        return RuleSet(combined.values())

    def without(self, *rule_ids: str) -> RuleSet:
        return RuleSet(r for r in self if r.id not in rule_ids)

    def match(self, text: str) -> list[RuleMatch]:
        return [
            RuleMatch(rule, start, end)
            for rule in self._rules.values()
            for start, end in rule.finditer(text)
        ]


def load_rules(*paths: str | Path, include_default: bool = True) -> RuleSet:
    """Load the built-in pack (optionally) plus any custom YAML packs."""
    ruleset = RuleSet.default() if include_default else RuleSet()
    for path in paths:
        ruleset = ruleset.merged(RuleSet.from_file(path))
    return ruleset

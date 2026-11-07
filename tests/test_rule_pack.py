"""The default rule pack's examples are executable documentation."""

import pytest

from promptfence.rules import Rule, RuleError, RuleSet, load_rules
from promptfence.types import Severity

PACK = RuleSet.default()
RULES = list(PACK)


def test_pack_is_non_trivial():
    assert len(PACK) >= 30
    assert len({r.category for r in PACK}) >= 9


@pytest.mark.parametrize("rule", RULES, ids=lambda r: r.id)
def test_rule_is_documented(rule: Rule):
    assert rule.examples, f"{rule.id} has no examples"
    assert rule.counter_examples, f"{rule.id} has no counter-examples"
    assert len(rule.explanation) > 40


@pytest.mark.parametrize(
    ("rule", "example"),
    [(r, e) for r in RULES for e in r.examples],
    ids=lambda x: x.id if isinstance(x, Rule) else x[:40],
)
def test_examples_match(rule: Rule, example: str):
    assert rule.matches(example)


@pytest.mark.parametrize(
    ("rule", "example"),
    [(r, e) for r in RULES for e in r.counter_examples],
    ids=lambda x: x.id if isinstance(x, Rule) else x[:40],
)
def test_counter_examples_do_not_match(rule: Rule, example: str):
    assert not rule.matches(example)


MINIMAL = """
version: 1
rules:
  - id: ACME-001
    name: say-the-word
    category: custom
    severity: low
    explanation: test rule
    patterns: ['\\bbanana\\b']
"""


def test_custom_pack_loads():
    rs = RuleSet.from_yaml(MINIMAL)
    rule = rs.get("ACME-001")
    assert rule.severity is Severity.LOW
    assert rule.matches("I like BANANA bread")


@pytest.mark.parametrize(
    ("patch", "message"),
    [
        (("severity: low", "severity: extreme"), "unknown severity"),
        (("ACME-001", "acme 1"), "invalid rule id"),
        (("['\\bbanana\\b']", "['(unclosed']"), "bad pattern"),
        (
            ("    explanation: test rule\n", "    explanation: test rule\n    colour: red\n"),
            "unknown field",
        ),
        (("    name: say-the-word\n", ""), "missing required"),
        (("version: 1", "version: 9"), "unsupported"),
    ],
)
def test_validation_errors(patch, message):
    old, new = patch
    assert old in MINIMAL
    with pytest.raises(RuleError, match=message):
        RuleSet.from_yaml(MINIMAL.replace(old, new))


def test_duplicate_ids_rejected():
    body = MINIMAL.split("rules:\n")[1]
    with pytest.raises(RuleError, match="duplicate"):
        RuleSet.from_yaml(MINIMAL + body)


def test_invalid_yaml_rejected():
    with pytest.raises(RuleError):
        RuleSet.from_yaml("rules: [unclosed")


def test_load_rules_merges_and_overrides(tmp_path):
    override = MINIMAL.replace("ACME-001", "PF-OVR-001").replace("severity: low", "severity: info")
    path = tmp_path / "custom.yaml"
    path.write_text(override)
    rs = load_rules(path)
    assert len(rs) == len(PACK)
    assert rs.get("PF-OVR-001").severity is Severity.INFO
    assert len(load_rules(path, include_default=False)) == 1


def test_without():
    assert "PF-OVR-001" not in PACK.without("PF-OVR-001")

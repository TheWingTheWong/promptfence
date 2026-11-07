"""Command-line interface.

    promptfence scan page.html email.eml          # human-readable report
    promptfence scan --json - < tool_output.txt   # machine-readable
    promptfence sanitize page.html -o clean.html
    promptfence rules --category jailbreak

Exit status: 0 if nothing reached ``--fail-on``, 1 if something did, 2 on
usage or I/O errors.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

from . import __version__
from .detector import Detector
from .rules import RuleError, load_rules
from .sanitize import Mode, Scope, sanitize
from .types import Layer, Report, Severity, Verdict

_SEVERITY_COLOR = {
    Severity.CRITICAL: "1;97;41",
    Severity.HIGH: "1;31",
    Severity.MEDIUM: "33",
    Severity.LOW: "36",
    Severity.INFO: "2",
}
_VERDICT_COLOR = {
    Verdict.MALICIOUS: "1;97;41",
    Verdict.SUSPICIOUS: "1;30;43",
    Verdict.BENIGN: "1;30;42",
}


class Style:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def __call__(self, text: str, code: str) -> str:
        return f"\x1b[{code}m{text}\x1b[0m" if self.enabled else text

    def dim(self, text: str) -> str:
        return self(text, "2")

    def bold(self, text: str) -> str:
        return self(text, "1")


def _use_color(choice: str, stream: TextIO) -> bool:
    if choice == "always":
        return True
    if choice == "never" or os.environ.get("NO_COLOR"):
        return False
    return hasattr(stream, "isatty") and stream.isatty()


def _read(path: str) -> str:
    if path == "-":
        return sys.stdin.read()
    return Path(path).read_bytes().decode("utf-8", errors="replace")


def _risk_bar(risk: float, width: int = 20) -> str:
    filled = round(risk * width)
    return "█" * filled + "░" * (width - filled)


def _line_col(text: str, offset: int) -> tuple[int, int]:
    line = text.count("\n", 0, offset) + 1
    col = offset - (text.rfind("\n", 0, offset) + 1) + 1
    return line, col


def render_report(name: str, report: Report, style: Style, *, context: int = 1) -> str:
    out: list[str] = []
    verdict = style(f" {report.verdict.value.upper()} ", _VERDICT_COLOR[report.verdict])
    clf = "" if report.classifier_score is None else f"  classifier {report.classifier_score:.2f}"
    out.append(style.bold(name))
    out.append(
        f"  {verdict}  risk {report.risk:.2f} {style.dim(_risk_bar(report.risk))}"
        f"{clf}  {style.dim(f'{report.elapsed_ms:.1f} ms')}"
    )

    evidence = [f for f in report.findings if f.layer is not Layer.CLASSIFIER]
    findings = [
        f
        for f in report.findings
        if f.severity > Severity.INFO
        # A classifier passage that a rule already explains is just noise here.
        and not (
            f.layer is Layer.CLASSIFIER
            and any(e.start < f.end and f.start < e.end for e in evidence)
        )
    ]
    if findings:
        out.append("")
    for f in findings:
        line, col = _line_col(report.text, f.start)
        sev = style(f"{f.severity.label.upper():8s}", _SEVERITY_COLOR[f.severity])
        via = f"  via {', '.join(f.via)}" if f.via else ""
        out.append(f"  {sev} {f.rule_id:11s} {f.name}  {style.dim(f'{line}:{col}{via}')}")
        out.append(f"           {style.dim(repr(f.excerpt))}")
        if f.layer is not Layer.CLASSIFIER:
            out.append(f"           {f.explanation}")
    if report.signals:
        sig = ", ".join(f"{k} x{v}" for k, v in sorted(report.signals.items()))
        out.append("")
        out.append(f"  {style.dim('obfuscation signals:')} {sig}")

    spans = report.spans(Severity.LOW, include_classifier=True)
    if spans:
        out.append("")
        out.extend(_highlighted_lines(report.text, spans, style, context))
    return "\n".join(out)


def _highlighted_lines(
    text: str, spans: list[tuple[int, int]], style: Style, context: int, max_width: int = 160
) -> list[str]:
    lines = text.split("\n")
    starts = [0]
    for line in lines[:-1]:
        starts.append(starts[-1] + len(line) + 1)
    hot = {i for i, s in enumerate(starts) for a, b in spans if a < s + len(lines[i]) + 1 and s < b}
    shown = sorted(
        {j for i in hot for j in range(i - context, i + context + 1) if 0 <= j < len(lines)}
    )
    width = len(str(len(lines)))
    out = []
    prev = None
    for i in shown:
        if prev is not None and i != prev + 1:
            out.append(style.dim(f"  {'':>{width}} ┆"))
        prev = i
        line, base = lines[i], starts[i]
        pieces = []
        cursor = 0
        for a, b in spans:
            a, b = max(a - base, 0), min(b - base, len(line))
            if a >= b or b <= cursor:
                continue
            a = max(a, cursor)
            pieces.append(line[cursor:a])
            pieces.append(style(line[a:b], "1;4;31") if style.enabled else f"»{line[a:b]}«")
            cursor = b
        pieces.append(line[cursor:])
        rendered = "".join(pieces)
        if len(line) > max_width and not style.enabled:
            rendered = rendered[: max_width + 2] + "…"
        out.append(f"  {style.dim(f'{i + 1:>{width}} │')} {rendered}")
    return out


def _build_detector(args: argparse.Namespace) -> Detector:
    rules = load_rules(*args.rules, include_default=not args.no_default_rules)
    return Detector(rules=rules, classifier=False if args.no_classifier else None)


def cmd_scan(args: argparse.Namespace) -> int:
    detector = _build_detector(args)
    style = Style(_use_color(args.color, sys.stdout))
    threshold = {"suspicious": Verdict.SUSPICIOUS, "malicious": Verdict.MALICIOUS}.get(args.fail_on)
    rank = {Verdict.BENIGN: 0, Verdict.SUSPICIOUS: 1, Verdict.MALICIOUS: 2}
    failed = False
    results = []
    for i, path in enumerate(args.paths or ["-"]):
        report = detector.scan(_read(path))
        if threshold is not None and rank[report.verdict] >= rank[threshold]:
            failed = True
        if args.json:
            results.append({"path": path, **report.to_dict()})
        elif args.quiet:
            print(f"{report.verdict.value:10s} {report.risk:.2f}  {path}")
        else:
            if i:
                print()
            print(render_report("<stdin>" if path == "-" else path, report, style,
                                context=args.context))  # fmt: skip
    if args.json:
        payload = results[0] if len(results) == 1 else results
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 1 if failed else 0


def cmd_sanitize(args: argparse.Namespace) -> int:
    detector = _build_detector(args)
    text = _read(args.path)
    cleaned = sanitize(
        text,
        detector.scan(text),
        mode=args.mode,
        scope=args.scope,
        min_severity=args.min_severity,
        include_classifier=args.include_classifier,
    )
    if args.output:
        Path(args.output).write_text(cleaned, encoding="utf-8")
    else:
        sys.stdout.write(cleaned)
    return 0


def cmd_rules(args: argparse.Namespace) -> int:
    rules = load_rules(*args.rules, include_default=not args.no_default_rules)
    style = Style(_use_color(args.color, sys.stdout))
    selected = [r for r in rules if not args.category or r.category == args.category]
    if args.json:
        print(json.dumps([{"id": r.id, "name": r.name, "category": r.category,
                           "severity": r.severity.label, "explanation": r.explanation}
                          for r in selected], indent=2))  # fmt: skip
        return 0
    for r in selected:
        sev = style(f"{r.severity.label:8s}", _SEVERITY_COLOR[r.severity])
        print(f"{r.id:11s} {sev} {r.category:24s} {r.name}")
        if args.verbose:
            print(f"            {r.explanation}")
    print(style.dim(f"\n{len(selected)} rule(s)"))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="promptfence",
        description="Detect prompt-injection and jailbreak attempts in untrusted text.",
    )
    parser.add_argument("--version", action="version", version=f"promptfence {__version__}")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--rules",
        action="append",
        default=[],
        metavar="YAML",
        help="extra rule pack (repeatable); same ids override built-ins",
    )
    common.add_argument(
        "--no-default-rules", action="store_true", help="do not load the built-in rule pack"
    )
    common.add_argument("--color", choices=["auto", "always", "never"], default="auto")
    sub = parser.add_subparsers(dest="command", required=True)

    scan = sub.add_parser("scan", parents=[common], help="scan files (or - for stdin)")
    scan.add_argument("paths", nargs="*", help="files to scan; default reads stdin")
    scan.add_argument("--json", action="store_true", help="emit JSON")
    scan.add_argument("-q", "--quiet", action="store_true", help="one line per file")
    scan.add_argument("--no-classifier", action="store_true", help="rules and normalizer only")
    scan.add_argument(
        "--fail-on",
        choices=["suspicious", "malicious", "never"],
        default="malicious",
        help="verdict that yields exit status 1",
    )
    scan.add_argument("--context", type=int, default=1, help="lines of context around hits")
    scan.set_defaults(func=cmd_scan)

    san = sub.add_parser("sanitize", parents=[common], help="neutralize detected spans")
    san.add_argument("path", help="file to sanitize, or - for stdin")
    san.add_argument("-o", "--output", help="write here instead of stdout")
    san.add_argument("--mode", choices=[m.value for m in Mode], default=Mode.REDACT.value)
    san.add_argument("--scope", choices=[s.value for s in Scope], default=Scope.SENTENCE.value)
    san.add_argument("--min-severity", choices=[s.label for s in Severity], default="medium")
    san.add_argument("--include-classifier", action="store_true")
    san.add_argument("--no-classifier", action="store_true", help="rules and normalizer only")
    san.set_defaults(func=cmd_sanitize)

    rules = sub.add_parser("rules", parents=[common], help="list rules")
    rules.add_argument("--category", help="only rules in this category")
    rules.add_argument("-v", "--verbose", action="store_true", help="include explanations")
    rules.add_argument("--json", action="store_true")
    rules.set_defaults(func=cmd_rules)
    return parser  # fmt: skip


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (OSError, RuleError) as exc:
        print(f"promptfence: error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

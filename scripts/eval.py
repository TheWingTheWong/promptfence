"""Evaluate promptfence on the held-out test split.

    uv run python scripts/eval.py                 # test split, summary table
    uv run python scripts/eval.py --errors        # also list misclassified examples
    uv run python scripts/eval.py --split train   # sanity check on training data

Reports each layer on its own and the combined detector, so the value of
every layer is visible rather than asserted.
"""

from __future__ import annotations

import argparse
import statistics
from collections import defaultdict
from pathlib import Path

from promptfence import Detector, Severity, Verdict
from promptfence.evaluation import Metrics, confusion, load_dataset
from promptfence.types import Layer

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "dataset.jsonl")
    parser.add_argument("--split", choices=["train", "test", "all"], default="test")
    parser.add_argument("--errors", action="store_true", help="list misclassified examples")
    args = parser.parse_args()

    examples = load_dataset(args.data, None if args.split == "all" else args.split)
    labels = [ex.label for ex in examples]
    detector = Detector()
    reports = [detector.scan(ex.text) for ex in examples]

    def rules_flag(r):
        return any(
            f.layer is not Layer.CLASSIFIER and f.severity >= Severity.MEDIUM for f in r.findings
        )

    predictions = {
        "rules + normalizer only (any finding >= medium)": map(rules_flag, reports),
        "classifier only (p >= 0.5)": (r.classifier_score >= 0.5 for r in reports),
        "combined, verdict >= suspicious": (r.is_suspicious for r in reports),
        "combined, verdict == malicious": (r.is_malicious for r in reports),
    }
    rows: list[tuple[str, Metrics]] = [
        (name, confusion(labels, preds)) for name, preds in predictions.items()
    ]

    pos, neg = sum(labels), len(labels) - sum(labels)
    print(f"split={args.split}: {len(examples)} examples ({pos} injection, {neg} benign)\n")
    print(f"| {'detector':50s} | precision | recall |   F1  |  FPR  |")
    print(f"|{'-' * 52}|-----------|--------|-------|-------|")
    for name, m in rows:
        print(
            f"| {name:50s} |   {m.precision:.3f}   | {m.recall:.3f}  | {m.f1:.3f} | "
            f"{m.false_positive_rate:.3f} |"
        )

    by_cat: dict[str, list[bool]] = defaultdict(list)
    for ex, r in zip(examples, reports, strict=True):
        by_cat[ex.category].append(r.is_malicious if ex.label else not r.is_suspicious)
    print("\nper-category accuracy (injection: flagged malicious; benign: verdict benign)")
    for ex_label, title in ((1, "injection"), (0, "benign")):
        cats = sorted({ex.category for ex in examples if ex.label == ex_label})
        parts = [f"{c} {sum(by_cat[c])}/{len(by_cat[c])}" for c in cats]
        print(f"  {title:9s}: " + ", ".join(parts))

    times = sorted(r.elapsed_ms for r in reports)
    p95 = times[int(0.95 * (len(times) - 1))]
    print(f"\nlatency: median {statistics.median(times):.2f} ms, p95 {p95:.2f} ms per document")

    if args.errors:
        print("\nmisclassified:")
        for ex, r in zip(examples, reports, strict=True):
            wrong = (ex.label == 1 and not r.is_malicious) or (ex.label == 0 and r.is_suspicious)
            if not wrong:
                continue
            kind = "FN" if ex.label else "FP"
            ids = ",".join(sorted({f.rule_id for f in r.findings})) or "-"
            verdict = r.verdict.value if r.verdict is not Verdict.BENIGN else "benign"
            clf = r.classifier_score or 0.0
            print(f"  {kind} [{ex.category}] risk={r.risk:.2f} {verdict} clf={clf:.2f} {ids}")
            print(f"     {ex.text[:110]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

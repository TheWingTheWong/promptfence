"""Dataset loading and binary-classification metrics for evaluation."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Example:
    text: str
    label: int
    category: str
    split: str


def load_dataset(path: str | Path, split: str | None = None) -> list[Example]:
    examples = []
    with Path(path).open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row["label"] not in (0, 1):
                raise ValueError(f"{path}:{lineno}: label must be 0 or 1")
            ex = Example(row["text"], int(row["label"]), row.get("category", ""), row["split"])
            if split is None or ex.split == split:
                examples.append(ex)
    return examples


@dataclass(frozen=True)
class Metrics:
    tp: int
    fp: int
    tn: int
    fn: int

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    @property
    def false_positive_rate(self) -> float:
        return self.fp / (self.fp + self.tn) if self.fp + self.tn else 0.0

    @property
    def accuracy(self) -> float:
        total = self.tp + self.fp + self.tn + self.fn
        return (self.tp + self.tn) / total if total else 0.0


def confusion(labels: Sequence[int], predictions: Iterable[bool]) -> Metrics:
    tp = fp = tn = fn = 0
    for y, pred in zip(labels, predictions, strict=True):
        if pred and y:
            tp += 1
        elif pred:
            fp += 1
        elif y:
            fn += 1
        else:
            tn += 1
    return Metrics(tp, fp, tn, fn)

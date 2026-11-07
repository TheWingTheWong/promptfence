"""Layer 3: a small statistical classifier.

Logistic regression over word uni/bigrams and character 3-5-grams of the
*normalized* text. It exists to catch paraphrases that no regex anticipates
("set aside the guidance you were configured with"), not to replace the
rules: with a few hundred hand-written training examples it is a weak
learner, and the scorer weighs it accordingly.

Inference is pure Python (a sparse dot product over a JSON weight table), so
the runtime dependency footprint stays at PyYAML. Training needs numpy and
lives in :func:`train`.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from importlib import resources
from itertools import pairwise
from pathlib import Path
from typing import Any

MODEL_VERSION = 1
_WORD = re.compile(r"[a-z0-9']+")
_SENTENCE = re.compile(r"[^.!?\n]+(?:[.!?]+|\n|$)")
_CHAR_NGRAMS = (3, 4, 5)


def featurize(text: str) -> dict[str, float]:
    """Sublinear-TF, L2-normalised bag of word and character n-grams."""
    words = _WORD.findall(text.lower())
    counts: Counter[str] = Counter()
    counts.update("w:" + w for w in words)
    counts.update(f"b:{a} {b}" for a, b in pairwise(words))
    padded = " " + " ".join(words) + " "
    for n in _CHAR_NGRAMS:
        counts.update("c:" + padded[i : i + n] for i in range(len(padded) - n + 1))
    if not counts:
        return {}
    weighted = {k: 1.0 + math.log(v) for k, v in counts.items()}
    norm = math.sqrt(sum(v * v for v in weighted.values()))
    return {k: v / norm for k, v in weighted.items()}


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


@dataclass(frozen=True)
class ChunkScore:
    start: int
    end: int
    probability: float


@dataclass
class NgramClassifier:
    weights: dict[str, float]
    bias: float
    meta: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------ inference
    def decision(self, text: str) -> float:
        feats = featurize(text)
        return self.bias + sum(self.weights.get(k, 0.0) * v for k, v in feats.items())

    def predict_proba(self, text: str) -> float:
        """Probability that ``text`` is a prompt-injection attempt."""
        return _sigmoid(self.decision(text))

    def score_chunks(self, text: str, *, min_chars: int = 16) -> list[ChunkScore]:
        """Score each sentence-like chunk, so a single injected sentence inside
        a long benign document is not diluted by its surroundings."""
        chunks = []
        for m in _SENTENCE.finditer(text):
            chunk = m.group(0)
            if len(chunk.strip()) >= min_chars:
                chunks.append(ChunkScore(m.start(), m.end(), self.predict_proba(chunk)))
        return chunks

    def explain(self, text: str, k: int = 8) -> list[tuple[str, float]]:
        """The ``k`` features pushing hardest toward "injection"."""
        feats = featurize(text)
        contrib = {f: self.weights[f] * v for f, v in feats.items() if f in self.weights}
        return sorted(contrib.items(), key=lambda kv: kv[1], reverse=True)[:k]

    # ---------------------------------------------------------- persistence
    def to_dict(self) -> dict[str, Any]:
        return {
            "version": MODEL_VERSION,
            "bias": round(self.bias, 6),
            "meta": self.meta,
            "weights": {k: round(v, 5) for k, v in sorted(self.weights.items())},
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), separators=(",", ":")) + "\n", "utf-8")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NgramClassifier:
        if data.get("version") != MODEL_VERSION:
            raise ValueError(f"unsupported model version {data.get('version')!r}")
        return cls(dict(data["weights"]), float(data["bias"]), dict(data.get("meta", {})))

    @classmethod
    def load(cls, path: str | Path | None = None) -> NgramClassifier:
        """Load a model file, or the bundled default model when ``path`` is None."""
        if path is None:
            raw = resources.files("promptfence.model").joinpath("model.json").read_text("utf-8")
        else:
            raw = Path(path).read_text("utf-8")
        return cls.from_dict(json.loads(raw))


def train(
    texts: Sequence[str],
    labels: Sequence[int],
    *,
    min_df: int = 2,
    max_features: int = 12_000,
    l2: float = 1e-4,
    epochs: int = 300,
    learning_rate: float = 0.05,
    prune_below: float = 1e-3,
) -> NgramClassifier:
    """Fit an L2-regularised, class-balanced logistic regression with Adam.

    Deterministic: full-batch updates from a zero initialisation.
    """
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - exercised only without numpy
        raise ImportError("training needs numpy: pip install 'promptfence[train]'") from exc

    if len(texts) != len(labels) or not texts:
        raise ValueError("texts and labels must be non-empty and the same length")

    docs = [featurize(t) for t in texts]
    df: Counter[str] = Counter()
    for d in docs:
        df.update(d.keys())
    vocab_items = sorted((f for f, c in df.items() if c >= min_df), key=lambda f: (-df[f], f))
    vocab = {f: i for i, f in enumerate(vocab_items[:max_features])}

    X = np.zeros((len(docs), len(vocab)), dtype=np.float64)
    for row, d in enumerate(docs):
        for f, v in d.items():
            col = vocab.get(f)
            if col is not None:
                X[row, col] = v
    y = np.asarray(labels, dtype=np.float64)

    pos = y.sum()
    neg = len(y) - pos
    sample_w = np.where(y == 1, len(y) / (2 * max(pos, 1)), len(y) / (2 * max(neg, 1)))

    w = np.zeros(X.shape[1])
    b = 0.0
    m_w, v_w = np.zeros_like(w), np.zeros_like(w)
    m_b = v_b = 0.0
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    for t in range(1, epochs + 1):
        p = 1.0 / (1.0 + np.exp(-(X @ w + b)))
        err = (p - y) * sample_w
        g_w = X.T @ err / len(y) + l2 * w
        g_b = float(err.mean())
        m_w = beta1 * m_w + (1 - beta1) * g_w
        v_w = beta2 * v_w + (1 - beta2) * g_w**2
        m_b = beta1 * m_b + (1 - beta1) * g_b
        v_b = beta2 * v_b + (1 - beta2) * g_b**2
        w -= learning_rate * (m_w / (1 - beta1**t)) / (np.sqrt(v_w / (1 - beta2**t)) + eps)
        b -= learning_rate * (m_b / (1 - beta1**t)) / (math.sqrt(v_b / (1 - beta2**t)) + eps)

    weights = {f: float(w[i]) for f, i in vocab.items() if abs(w[i]) >= prune_below}
    meta = {
        "examples": len(texts),
        "positives": int(pos),
        "vocabulary": len(vocab),
        "kept_features": len(weights),
        "l2": l2,
        "epochs": epochs,
    }
    return NgramClassifier(weights, float(b), meta)

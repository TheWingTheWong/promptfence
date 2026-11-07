"""Train the n-gram classifier on the train split and write the bundled model.

    uv run python scripts/train.py

Only the ``train`` split is used; ``test`` is reserved for scripts/eval.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from promptfence.classifier import train
from promptfence.detector import classifier_text
from promptfence.evaluation import confusion, load_dataset
from promptfence.normalize import normalize

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "dataset.jsonl")
    parser.add_argument(
        "--out", type=Path, default=ROOT / "src" / "promptfence" / "model" / "model.json"
    )
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=300)
    args = parser.parse_args()

    examples = load_dataset(args.data, split="train")
    texts = [classifier_text(normalize(ex.text)) for ex in examples]
    labels = [ex.label for ex in examples]
    model = train(texts, labels, l2=args.l2, epochs=args.epochs)

    fit = confusion(labels, (model.predict_proba(t) >= 0.5 for t in texts))
    print(f"trained on {len(examples)} examples, kept {len(model.weights)} features")
    print(f"train-set fit: precision={fit.precision:.3f} recall={fit.recall:.3f}")
    print("top injection features:")
    for feat, w in sorted(model.weights.items(), key=lambda kv: -kv[1])[:12]:
        print(f"  {w:+.3f}  {feat!r}")

    model.save(args.out)
    print(f"wrote {args.out.relative_to(ROOT)} ({args.out.stat().st_size // 1024} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

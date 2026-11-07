"""Build data/dataset.jsonl from the hand-written sources in data/src/.

Source format: one example per line, ``\\n`` for a newline inside an example,
and ``## <category>`` headers. Obfuscated attack examples are written as
templates so the encoded form is reproducible and the plain intent stays
reviewable in the source:

    {{b64:...}}   base64            {{rot13:...}}  ROT13
    {{hex:...}}   space-separated hex bytes
    {{tags:...}}  Unicode tag characters (invisible)
    {{zw:...}}    zero-width spaces between letters
    {{homo:...}}  Cyrillic homoglyphs swapped in for Latin letters

The train/test split is assigned by a hash of the final text, so it is stable
across rebuilds and independent of file order.
"""

from __future__ import annotations

import base64
import codecs
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCES = {"benign": 0, "injection": 1}
TEST_FRACTION = 0.25

_HOMO = {"a": "а", "e": "е", "o": "о", "i": "і", "c": "с", "p": "р"}
_TEMPLATE = re.compile(r"\{\{(b64|rot13|hex|tags|zw|homo):(.*?)\}\}")


def _homoglyphs(s: str) -> str:
    out, n = [], 0
    for ch in s:
        if ch in _HOMO:
            n += 1
            out.append(_HOMO[ch] if n % 2 else ch)
        else:
            out.append(ch)
    return "".join(out)


_RENDER = {
    "b64": lambda s: base64.b64encode(s.encode()).decode(),
    "rot13": lambda s: codecs.encode(s, "rot13"),
    "hex": lambda s: " ".join(f"{b:02x}" for b in s.encode()),
    "tags": lambda s: "".join(chr(0xE0000 + ord(c)) for c in s),
    "zw": lambda s: "​".join(s),
    "homo": _homoglyphs,
}


def render(line: str) -> str:
    line = line.replace("\\n", "\n")
    return _TEMPLATE.sub(lambda m: _RENDER[m.group(1)](m.group(2)), line)


def split_for(text: str) -> str:
    bucket = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "test" if bucket < TEST_FRACTION else "train"


def main() -> int:
    rows = []
    for name, label in SOURCES.items():
        category = "uncategorized"
        for raw in (ROOT / "data" / "src" / f"{name}.txt").read_text("utf-8").splitlines():
            if not raw.strip():
                continue
            if raw.startswith("## "):
                category = raw[3:].strip()
                continue
            text = render(raw)
            rows.append(
                {"text": text, "label": label, "category": category, "split": split_for(text)}
            )
    seen: set[str] = set()
    for row in rows:
        if row["text"] in seen:
            print(f"duplicate example: {row['text'][:60]!r}", file=sys.stderr)
            return 1
        seen.add(row["text"])
    out = ROOT / "data" / "dataset.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    for split in ("train", "test"):
        pos = sum(r["split"] == split and r["label"] == 1 for r in rows)
        neg = sum(r["split"] == split and r["label"] == 0 for r in rows)
        print(f"{split:5s}: {pos:4d} injection  {neg:4d} benign")
    print(f"wrote {len(rows)} examples to {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

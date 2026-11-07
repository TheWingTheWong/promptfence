"""Offset-preserving text transforms.

Normalization rewrites the input (folds homoglyphs, strips tags, decodes
payloads...), but every finding must point back at the *original* characters
so callers can highlight or sanitize them. :class:`MappedText` carries, for
each character of the transformed text, the ``[start, end)`` range of the
original text it came from.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

Replacement = str | Callable[[re.Match[str]], str]


@dataclass(frozen=True)
class MappedText:
    text: str
    starts: tuple[int, ...]
    ends: tuple[int, ...]

    def __post_init__(self) -> None:
        if not (len(self.text) == len(self.starts) == len(self.ends)):
            raise ValueError("text and offset maps must have equal length")

    @classmethod
    def identity(cls, text: str, offset: int = 0) -> MappedText:
        positions = range(offset, offset + len(text))
        return cls(text, tuple(positions), tuple(p + 1 for p in positions))

    @classmethod
    def constant(cls, text: str, start: int, end: int) -> MappedText:
        """Every character maps to the same original range (e.g. a decoded blob)."""
        n = len(text)
        return cls(text, (start,) * n, (end,) * n)

    def __len__(self) -> int:
        return len(self.text)

    def span(self, start: int, end: int) -> tuple[int, int]:
        """Map a ``[start, end)`` range of this text to the original text."""
        if not self.text:
            return (0, 0)
        start = max(0, min(start, len(self.text) - 1))
        end = max(start + 1, min(end, len(self.text)))
        return (min(self.starts[start:end]), max(self.ends[start:end]))

    def map_chars(self, fn: Callable[[str], str]) -> MappedText:
        """Apply ``fn`` to every character; output characters inherit its mapping."""
        out: list[str] = []
        starts: list[int] = []
        ends: list[int] = []
        for ch, s, e in zip(self.text, self.starts, self.ends, strict=True):
            rep = fn(ch)
            if rep:
                out.append(rep)
                starts.extend([s] * len(rep))
                ends.extend([e] * len(rep))
        return MappedText("".join(out), tuple(starts), tuple(ends))

    def sub(self, pattern: re.Pattern[str] | str, repl: Replacement) -> MappedText:
        """Regex substitution that keeps the offset maps consistent.

        If a replacement has the same length as its match, characters map
        one-to-one; otherwise every replacement character maps to the whole
        original range covered by the match.
        """
        regex = re.compile(pattern) if isinstance(pattern, str) else pattern
        out: list[str] = []
        starts: list[int] = []
        ends: list[int] = []
        cursor = 0
        changed = False
        for m in regex.finditer(self.text):
            a, b = m.span()
            rep = repl(m) if callable(repl) else m.expand(repl)
            if rep == m.group(0):
                continue
            changed = True
            out.append(self.text[cursor:a])
            starts.extend(self.starts[cursor:a])
            ends.extend(self.ends[cursor:a])
            if len(rep) == b - a:
                starts.extend(self.starts[a:b])
                ends.extend(self.ends[a:b])
            elif rep:
                if a == b:  # zero-width match: anchor to the neighbouring char
                    anchor = min(a, len(self.text) - 1)
                    lo, hi = self.starts[anchor], self.ends[anchor]
                else:
                    lo, hi = min(self.starts[a:b]), max(self.ends[a:b])
                starts.extend([lo] * len(rep))
                ends.extend([hi] * len(rep))
            out.append(rep)
            cursor = b
        if not changed:
            return self
        out.append(self.text[cursor:])
        starts.extend(self.starts[cursor:])
        ends.extend(self.ends[cursor:])
        return MappedText("".join(out), tuple(starts), tuple(ends))

    def compose(self, outer: MappedText) -> MappedText:
        """Re-express offsets that point into ``outer.text`` as offsets into
        whatever ``outer`` itself maps to."""
        if not outer.text:
            return MappedText(self.text, (0,) * len(self), (0,) * len(self))
        last = len(outer.text) - 1
        starts = tuple(outer.starts[min(s, last)] for s in self.starts)
        ends = tuple(outer.ends[min(max(e - 1, 0), last)] for e in self.ends)
        return MappedText(self.text, starts, ends)

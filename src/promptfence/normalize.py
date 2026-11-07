"""Layer 1: normalization.

Turns raw untrusted input into one or more *views* that the rule engine and
classifier can match against, while recording every obfuscation trick it had
to undo (an *artifact*). Every view keeps an offset map back to the original
input, so findings can always be highlighted in the text the caller passed in.

Views produced:

``canonical``
    Invisible characters removed, markup stripped (comment and hidden-element
    text is kept, attribute text like ``alt=`` is surfaced), entities decoded,
    compatibility forms and diacritics folded, mixed-script homoglyphs folded
    to ASCII, and s p a c e d - o u t words joined.
``markup``
    Only for HTML/Markdown input: the source with invisible characters
    removed but tags intact, for rules that look at markup itself
    (``<img src=...>`` beacons, ``<<SYS>>`` delimiters).
``deobfuscated``
    ``canonical`` plus leetspeak folding (``1gn0r3`` -> ``ignore``). Kept
    separate because folding digits would damage URLs and numbers.
``base64`` / ``hex`` / ``rot13`` / ``url-encoding`` / ``unicode-escape`` / ``unicode-tags``
    Decoded payloads, themselves normalized recursively. Their offsets point
    at the encoded blob in the original text.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field

from .confusables import CONFUSABLES, LEET
from .decoders import find_encoded
from .mapped import MappedText
from .markup import attribute_texts, find_hidden_regions, looks_like_markup, strip_markup

# Zero-width, joiners, bidi controls, soft hyphen, word joiner, BOM, variation selectors.
_INVISIBLE = re.compile("[­͏؜ᅟᅠ឴឵᠋-᠎​-‏‪-‮⁠-⁤⁦-⁯ㅤ︀-️﻿ﾠ\U0001d173-\U0001d17a\U000e0100-\U000e01ef]+")
# Unicode "tag" characters mirror ASCII and render as nothing: a known channel
# for smuggling instructions past human reviewers.
_TAG_RUN = re.compile("[\U000e0000-\U000e007f]+")
_BLACK_FLAG = "\U0001f3f4"  # subdivision flags (England, Scotland...) legitimately use tags
_LETTER_TOKEN = re.compile(r"[^\W\d_]+")
_LEET_TOKEN = re.compile(r"(?<![\w@$])[A-Za-z0-9@$]{3,20}(?![\w@$])")
_SPACED = re.compile(r"(?<!\w)(?:[A-Za-z][ \t.\-_*·•|]{1,2}){4,}[A-Za-z](?!\w)")
_SPACED_SEP = re.compile(r"[ \t.\-_*·•|]")

HIDDEN_KINDS = frozenset({"html-comment", "markdown-comment", "css-hidden", "html-attribute"})
ENCODED_KINDS = frozenset(
    {"base64", "hex", "rot13", "url-encoding", "unicode-escape", "unicode-tags"}
)


@dataclass(frozen=True)
class View:
    name: str
    mapped: MappedText
    #: Obfuscation layers peeled off to reach this view, outermost first.
    via: tuple[str, ...] = ()

    @property
    def text(self) -> str:
        return self.mapped.text


@dataclass(frozen=True)
class Artifact:
    """Evidence of an obfuscation technique, anchored in the original text."""

    kind: str
    start: int
    end: int
    detail: str = ""


@dataclass
class Normalized:
    original: str
    views: list[View]
    artifacts: list[Artifact] = field(default_factory=list)

    @property
    def canonical(self) -> View:
        return self.views[0]

    @property
    def signals(self) -> dict[str, int]:
        return dict(Counter(a.kind for a in self.artifacts))

    def hidden_kind(self, start: int, end: int) -> str | None:
        """If ``[start, end)`` lies inside human-invisible content, say which kind."""
        for a in self.artifacts:
            if a.kind in HIDDEN_KINDS and a.start <= start and end <= a.end:
                return a.kind
        return None


def fold_char(ch: str) -> str:
    """Fold compatibility forms and strip diacritics: ``ｉ`` -> ``i``, ``é`` -> ``e``."""
    if ch.isascii():
        return ch
    decomposed = unicodedata.normalize("NFKD", ch)
    base = "".join(c for c in decomposed if not unicodedata.combining(c))
    return unicodedata.normalize("NFKC", base) if base else ""


def _fold_homoglyphs(view: MappedText, artifacts: list[Artifact]) -> MappedText:
    def fold(m: re.Match[str]) -> str:
        token = m.group(0)
        if not any(ch in CONFUSABLES for ch in token):
            return token
        has_ascii = any(ch.isascii() for ch in token)
        all_confusable = all(ch.isascii() or ch in CONFUSABLES for ch in token)
        if all_confusable and (has_ascii or len(token) > 1):
            if has_ascii:  # mixed script inside one word is the actual red flag
                artifacts.append(Artifact("homoglyph", *view.span(*m.span()), detail=token))
            return "".join(CONFUSABLES.get(ch, ch) for ch in token)
        return token

    return view.sub(_LETTER_TOKEN, fold)


def _join_spaced(view: MappedText, artifacts: list[Artifact]) -> MappedText:
    def join(m: re.Match[str]) -> str:
        artifacts.append(Artifact("spaced-letters", *view.span(*m.span()), detail=m.group(0)))
        return _SPACED_SEP.sub("", m.group(0))

    return view.sub(_SPACED, join)


def _fold_leet(view: MappedText, artifacts: list[Artifact]) -> MappedText:
    def fold(m: re.Match[str]) -> str:
        token = m.group(0)
        letters = sum(ch.isalpha() for ch in token)
        subs = sum(ch in LEET for ch in token)
        if subs == 0 or letters < 2 or subs > letters or token.isdigit():
            return token
        folded = "".join(LEET.get(ch, ch) for ch in token)
        artifacts.append(Artifact("leetspeak", *view.span(*m.span()), detail=token))
        return folded

    return view.sub(_LEET_TOKEN, fold)


def _decode_tag_run(run: str) -> str:
    return "".join(chr(ord(c) - 0xE0000) for c in run if 0xE0020 <= ord(c) <= 0xE007E)


def normalize(text: str, *, max_depth: int = 2, _depth: int = 0) -> Normalized:
    """Build the normalized views of ``text``. See the module docstring."""
    artifacts: list[Artifact] = []
    payloads: list[tuple[str, int, int, str]] = []  # kind, start, end, decoded text

    view = MappedText.identity(text)

    # 1. Unicode tag smuggling: decode, then drop from the visible text.
    for m in _TAG_RUN.finditer(text):
        if m.start() > 0 and text[m.start() - 1] == _BLACK_FLAG:
            continue
        decoded = _decode_tag_run(m.group(0))
        if decoded.strip():
            artifacts.append(Artifact("unicode-tags", m.start(), m.end(), detail=decoded[:80]))
            payloads.append(("unicode-tags", m.start(), m.end(), decoded))
    view = view.sub(_TAG_RUN, "")

    # 2. Invisible formatting characters.
    for m in _INVISIBLE.finditer(text):
        # Joiners and variation selectors inside emoji sequences are legitimate;
        # only flag invisibles that touch letters or digits.
        before = text[m.start() - 1] if m.start() > 0 else " "
        after = text[m.end()] if m.end() < len(text) else " "
        if before.isalnum() or after.isalnum():
            detail = f"{len(m.group(0))} invisible char(s)"
            artifacts.append(Artifact("invisible", m.start(), m.end(), detail=detail))
    view = view.sub(_INVISIBLE, "")

    # 3. Markup: remember what a reader would not see, then strip tags.
    markup_view: MappedText | None = None
    if looks_like_markup(text):
        markup_view = view
        for region in find_hidden_regions(text):
            artifacts.append(Artifact(region.kind, region.start, region.end))
        for start, end in attribute_texts(text):
            artifacts.append(Artifact("html-attribute", start, end))
        view = strip_markup(view)

    raw = view  # case and digits intact: what the decoders need

    # 4. Character folding for matching.
    view = view.map_chars(fold_char)
    view = _fold_homoglyphs(view, artifacts)
    view = _join_spaced(view, artifacts)
    views = [View("canonical", view)]
    if markup_view is not None:
        views.append(View("markup", markup_view))

    deobfuscated = _fold_leet(view, artifacts)
    if deobfuscated.text != view.text:
        views.append(View("deobfuscated", deobfuscated))

    # 5. Encoded payloads, normalized recursively.
    if _depth < max_depth:
        for d in find_encoded(raw.text):
            start, end = raw.span(d.start, d.end)
            artifacts.append(Artifact(d.kind, start, end, detail=d.text[:80]))
            payloads.append((d.kind, start, end, d.text))

        for kind, start, end, decoded in payloads:
            inner = normalize(decoded, max_depth=max_depth, _depth=_depth + 1)
            for v in inner.views:
                pinned = MappedText.constant(v.text, start, end)
                views.append(View(kind, pinned, (kind, *v.via)))
            artifacts.extend(
                Artifact(a.kind, start, end, a.detail)
                for a in inner.artifacts
                if a.kind not in HIDDEN_KINDS
            )

    artifacts.sort(key=lambda a: (a.start, a.end, a.kind))
    return Normalized(text, views, artifacts)

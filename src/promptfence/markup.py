"""Markup handling: find content a human reader would never see.

Indirect prompt injection usually hides in places a browser does not render:
HTML comments, elements styled out of view, ``alt``/``title`` attributes, or
Markdown comment syntax. We do not need a full HTML parser for this - we need
to (a) know where such regions are, so matches inside them can be escalated,
and (b) produce a tag-free text view whose offsets still point at the source.

The regexes here are intentionally tolerant: attacker-controlled markup is
often malformed, and a strict parser would simply give up on it.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass

from .mapped import MappedText

_HTML_HINT = re.compile(r"<(?:!--|/?[a-zA-Z][\w-]*(?:\s[^<>]*)?/?>)")
_COMMENT = re.compile(r"<!--(.*?)(?:-->|\Z)", re.S)
_MD_COMMENT = re.compile(
    r"^[ \t]*\[(?://|comment|_)\]:\s*(?:#|<>)\s*[(\"'](.*?)[)\"']?[ \t]*$", re.M
)
_OPEN_TAG = re.compile(r"<([a-zA-Z][\w-]*)\b([^<>]*?)(/?)>", re.S)
_ANY_TAG = re.compile(r"<!--|-->|<![^<>]*>|</?[a-zA-Z][\w-]*\b[^<>]*>", re.S)
_ENTITY = re.compile(r"&(?:#[0-9]{1,7}|#[xX][0-9a-fA-F]{1,6}|[a-zA-Z][a-zA-Z0-9]{1,31});")
_TEXT_ATTRS = re.compile(
    r"""\b(alt|title|aria-label|placeholder|content|data-[\w-]+)\s*=\s*("([^"]*)"|'([^']*)')""",
    re.I,
)
_HIDING_STYLE = re.compile(
    r"""display\s*:\s*none
      | visibility\s*:\s*hidden
      | font-size\s*:\s*0(?:\.0+)?(?:px|pt|em|rem|%)?\s*(?:;|$|!)
      | opacity\s*:\s*0(?:\.0+)?\s*(?:;|$|!)
      | color\s*:\s*(?:\#fff(?:fff)?|white|transparent|rgba\([^)]*,\s*0\))\b
      | (?:left|top|text-indent)\s*:\s*-\d{3,}
      | (?:height|width|max-height)\s*:\s*0(?:px)?\s*;[^"']*overflow\s*:\s*hidden
      | clip\s*:\s*rect\(\s*0""",
    re.I | re.X,
)
_HIDDEN_ATTR = re.compile(r"""(?:^|\s)(?:hidden|aria-hidden\s*=\s*["']?true)""", re.I)
_VOID_TAGS = frozenset("area base br col embed hr img input link meta source track wbr".split())


@dataclass(frozen=True)
class HiddenRegion:
    kind: str
    start: int  # inclusive, offset of the hidden *content*
    end: int


def looks_like_markup(text: str) -> bool:
    return bool(_HTML_HINT.search(text) or _MD_COMMENT.search(text))


def find_hidden_regions(text: str) -> list[HiddenRegion]:
    regions: list[HiddenRegion] = []
    for m in _COMMENT.finditer(text):
        if m.group(1).strip():
            regions.append(HiddenRegion("html-comment", m.start(1), m.end(1)))
    for m in _MD_COMMENT.finditer(text):
        if m.group(1).strip():
            regions.append(HiddenRegion("markdown-comment", m.start(1), m.end(1)))
    for m in _OPEN_TAG.finditer(text):
        tag, attrs, self_closing = m.group(1).lower(), m.group(2), m.group(3)
        if self_closing or tag in _VOID_TAGS:
            continue
        if not (_HIDING_STYLE.search(attrs) or _HIDDEN_ATTR.search(attrs)):
            continue
        close = _find_close(text, tag, m.end())
        if close > m.end():
            regions.append(HiddenRegion("css-hidden", m.end(), close))
    return sorted(regions, key=lambda r: (r.start, r.end))


def _find_close(text: str, tag: str, pos: int) -> int:
    """Return the start of the matching ``</tag>``, honouring nesting."""
    pattern = re.compile(rf"<(/?){re.escape(tag)}\b[^<>]*>", re.I)
    depth = 1
    for m in pattern.finditer(text, pos):
        depth += -1 if m.group(1) else 1
        if depth == 0:
            return m.start()
    return len(text)


def strip_markup(view: MappedText) -> MappedText:
    """Remove tags (keeping human-invisible attribute text) and decode entities."""

    def replace_tag(m: re.Match[str]) -> str:
        token = m.group(0)
        if token in ("<!--", "-->") or token.startswith("<!") or token.startswith("</"):
            return " "
        values = [
            a.group(3) if a.group(3) is not None else a.group(4)
            for a in _TEXT_ATTRS.finditer(token)
        ]
        values = [v for v in values if v and v.strip()]
        return " " + " ".join(values) + " " if values else " "

    stripped = view.sub(_ANY_TAG, replace_tag)
    return stripped.sub(_ENTITY, lambda m: html.unescape(m.group(0)))


def attribute_texts(text: str) -> list[tuple[int, int]]:
    """Spans of human-invisible attribute values (``alt``, ``title``...)."""
    spans = []
    for tag in _OPEN_TAG.finditer(text):
        for a in _TEXT_ATTRS.finditer(tag.group(0)):
            group = 3 if a.group(3) is not None else 4
            if a.group(group) and len(a.group(group).strip()) >= 20:
                spans.append((tag.start() + a.start(group), tag.start() + a.end(group)))
    return spans

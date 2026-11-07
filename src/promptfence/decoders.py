"""Find and decode encoded payloads hidden in otherwise ordinary text.

Attackers encode instructions (base64, hex, URL-encoding, ROT13, escape
sequences) so that keyword filters miss them while a capable model can still
decode and follow them. We decode candidates eagerly but only *keep* results
that look like natural language, so hashes, image blobs and tokens are
discarded instead of becoming noise.
"""

from __future__ import annotations

import base64
import binascii
import codecs
import re
from dataclasses import dataclass
from urllib.parse import unquote

_BASE64 = re.compile(r"(?<![A-Za-z0-9+/_-])[A-Za-z0-9+/_-]{16,}={0,2}(?![A-Za-z0-9+/=_-])")
_HEX_ESCAPED = re.compile(r"(?:\\x[0-9a-fA-F]{2}){6,}")
_HEX_PLAIN = re.compile(r"(?<![0-9a-fA-F])(?:[0-9a-fA-F]{2}[ :]?){8,}(?![0-9a-fA-F])")
_UNICODE_ESC = re.compile(r"(?:\\u[0-9a-fA-F]{4}){4,}")
_URL_ENC = re.compile(r"[^\s%]*(?:%[0-9A-Fa-f]{2}[^\s%]*){5,}")
_WORD = re.compile(r"[A-Za-z]+")

# A tiny English "fingerprint": frequent function words plus a few words that
# dominate instruction-like text. Used only to recognise ROT13'd English.
_COMMON_WORDS = frozenset(
    """the be to of and a in that have i it for not on with he as you do at this
    but his by from they we say her she or an will my one all would there their
    what so up out if about who get which go me when make can like time no just
    him know take people into year your good some could them see other than then
    now look only come its over think also back after use two how our work first
    well way even new want because any these give day most us is are was were
    ignore previous instructions instruction prompt system reveal print tell
    secret password rules above below following must should""".split()
)


@dataclass(frozen=True)
class Decoded:
    kind: str
    start: int
    end: int
    text: str


def looks_like_language(text: str, *, min_letters: int = 8) -> bool:
    """Heuristic: is ``text`` readable prose rather than binary or an identifier?"""
    if not text:
        return False
    printable = sum(ch.isprintable() or ch in "\n\r\t" for ch in text)
    if printable / len(text) < 0.95:
        return False
    letters = sum(ch.isalpha() for ch in text)
    if letters < min_letters or letters / len(text) < 0.55:
        return False
    return len(_WORD.findall(text)) >= 2 and " " in text.strip()


def _try_base64(token: str) -> str | None:
    stripped = token.rstrip("=")
    if len(set(stripped)) < 8:
        return None
    # Plain words and identifiers (all letters, single case) are not base64.
    if stripped.isalpha() and (stripped.islower() or stripped.isupper()):
        return None
    padded = stripped + "=" * (-len(stripped) % 4)
    altchars = b"-_" if ("-" in token or "_" in token) else None
    try:
        raw = base64.b64decode(padded, altchars=altchars, validate=True)
        return raw.decode("utf-8")
    except (binascii.Error, ValueError, UnicodeDecodeError):
        return None


def _try_hex(token: str) -> str | None:
    digits = re.sub(r"\\x|[ :]", "", token)
    if len(digits) % 2:
        digits = digits[:-1]
    try:
        return bytes.fromhex(digits).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return None


def _try_unicode_escape(token: str) -> str | None:
    try:
        return codecs.decode(token, "unicode_escape")
    except UnicodeDecodeError:
        return None


def _common_word_count(text: str) -> int:
    return sum(w.lower() in _COMMON_WORDS for w in _WORD.findall(text))


def _rot13_segments(text: str) -> list[Decoded]:
    found = []
    for m in re.finditer(r"[^\n]+", text):
        line = m.group(0)
        words = _WORD.findall(line)
        if len(words) < 4:
            continue
        plain = _common_word_count(line)
        rotated = codecs.encode(line, "rot13")
        decoded = _common_word_count(rotated)
        if decoded >= 3 and decoded >= 2 * plain + 2 and decoded / len(words) >= 0.3:
            found.append(Decoded("rot13", m.start(), m.end(), rotated))
    return found


def find_encoded(text: str) -> list[Decoded]:
    """Return every decodable payload in ``text`` that decodes to language."""
    found: list[Decoded] = []

    def keep(kind: str, m: re.Match[str], decoded: str | None) -> None:
        if decoded and decoded != m.group(0) and looks_like_language(decoded):
            found.append(Decoded(kind, m.start(), m.end(), decoded))

    for m in _BASE64.finditer(text):
        keep("base64", m, _try_base64(m.group(0)))
    for m in _HEX_ESCAPED.finditer(text):
        keep("hex", m, _try_hex(m.group(0)))
    for m in _HEX_PLAIN.finditer(text):
        keep("hex", m, _try_hex(m.group(0).strip()))
    for m in _UNICODE_ESC.finditer(text):
        keep("unicode-escape", m, _try_unicode_escape(m.group(0)))
    for m in _URL_ENC.finditer(text):
        keep("url-encoding", m, unquote(m.group(0)))
    found.extend(_rot13_segments(text))

    # Drop payloads fully contained in another payload of the same kind.
    found.sort(key=lambda d: (d.start, -(d.end - d.start)))
    result: list[Decoded] = []
    for d in found:
        if any(r.kind == d.kind and r.start <= d.start and d.end <= r.end for r in result):
            continue
        result.append(d)
    return result

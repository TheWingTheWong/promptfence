"""Homoglyph tables.

NFKC already folds fullwidth forms, mathematical alphanumerics (``𝐢𝐠𝐧𝐨𝐫𝐞``),
ligatures and most compatibility characters. What it does *not* fold are
letters from other scripts that merely look Latin - Cyrillic ``о``, Greek
``ο``, Armenian ``օ`` - which is the classic way to slip a phrase past a
keyword filter. This is a deliberately small, hand-curated subset of Unicode
TR39 ``confusables.txt`` covering lookalikes of ASCII letters.
"""

from __future__ import annotations

CONFUSABLES: dict[str, str] = {
    # Cyrillic lowercase
    "а": "a", "в": "b", "с": "c", "ԁ": "d", "е": "e", "ё": "e", "һ": "h",
    "і": "i", "ї": "i", "ј": "j", "к": "k", "ӏ": "l", "м": "m", "н": "h",
    "о": "o", "р": "p", "ԛ": "q", "ѕ": "s", "т": "t", "ц": "u", "ѵ": "v",
    "ԝ": "w", "х": "x", "у": "y", "ү": "y", "г": "r", "п": "n", "ь": "b",
    # Cyrillic uppercase
    "А": "A", "В": "B", "С": "C", "Е": "E", "Н": "H", "І": "I", "Ј": "J",
    "К": "K", "М": "M", "О": "O", "Р": "P", "Ѕ": "S", "Т": "T", "Х": "X",
    "У": "Y", "Ԝ": "W", "Ү": "Y",
    # Greek lowercase
    "α": "a", "β": "b", "ε": "e", "η": "n", "ι": "i", "κ": "k", "ν": "v",
    "ο": "o", "ρ": "p", "τ": "t", "υ": "u", "χ": "x", "γ": "y", "ϲ": "c",
    "ϳ": "j", "ω": "w",
    # Greek uppercase
    "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I", "Κ": "K",
    "Μ": "M", "Ν": "N", "Ο": "O", "Ρ": "P", "Τ": "T", "Υ": "Y", "Χ": "X",
    # Armenian
    "օ": "o", "ս": "u", "ո": "n", "հ": "h", "ց": "g", "ք": "p", "զ": "q",
    # Latin extensions / IPA that survive NFKC
    "ı": "i", "ȷ": "j", "ɡ": "g", "ɑ": "a", "ɩ": "i", "ʏ": "y", "ʀ": "r",
    "ɴ": "n", "ꞵ": "b", "ꮃ": "w", "ᴀ": "a", "ᴄ": "c", "ᴅ": "d", "ᴇ": "e",
    "ᴋ": "k", "ᴍ": "m", "ᴏ": "o", "ᴘ": "p", "ᴛ": "t", "ᴜ": "u", "ᴠ": "v",
    "ᴡ": "w", "ᴢ": "z",
}  # fmt: skip

# Digits and symbols commonly used as letter substitutes ("1gn0r3").
LEET: dict[str, str] = {
    "0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b",
    "@": "a", "$": "s", "!": "i", "|": "l",
}  # fmt: skip


def is_confusable(ch: str) -> bool:
    return ch in CONFUSABLES

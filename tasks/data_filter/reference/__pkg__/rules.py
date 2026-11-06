"""Blocklist rules: phrase matching and a density score.

A :class:`BlockList` holds a list of terms (single words or multi-word
phrases). Matching is case-insensitive and must respect word boundaries, so a
term never fires on a substring of a longer word (``"art"`` must not match
``"party"``). Terms may start or end with punctuation (``"c++"``, ``"u.s."``);
the boundary rule is the same for them: a match may not be preceded or
followed by a letter or digit.

Web text is messy, so both the terms and the document are normalised before
matching: Unicode NFKC (fullwidth / compatibility forms fold to their plain
equivalents), case-folded, and every run of whitespace (spaces, tabs,
newlines, no-break spaces) collapsed to a single space. ``"Free\\n  Ship"`` and
``"ＦＲＥＥ ＳＨＩＰ"`` therefore both match the term ``"free ship"``, and
``find_matches`` reports hits in that normalised form.

``tokenize`` applies the same normalisation and splits into words made of
Unicode letters and digits (apostrophes kept inside words), so non-English
text (``"café"``, ``"мир"``) is tokenized properly.

``score(text)`` is the match *density*: number of blocklist hits per 1,000
tokens, where tokens are produced by :func:`tokenize`. A document is a rule hit
when its density is at or above ``BlockList.threshold``.

Terms files (:meth:`BlockList.from_file`) are UTF-8, optionally with a BOM.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

TOKEN_RE = re.compile(r"(?:[^\W_]|')+")
_WS_RE = re.compile(r"\s+")

# Default terms for the __filter_noun__. The production list is longer; these
# are the development defaults used by the CLI when no --terms file is given.
DEFAULT_TERMS = [
    "__FlagTerm1__",
    "__FlagTerm2__",
    "__FlagTerm3__",
    "__FlagTerm4__",
]

DEFAULT_DENSITY_THRESHOLD = 10.0  # hits per 1k tokens


def normalize(text: str) -> str:
    """NFKC, case-fold and collapse whitespace runs to one space."""
    return _WS_RE.sub(" ", unicodedata.normalize("NFKC", text).casefold()).strip()


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(normalize(text))


class BlockList:
    def __init__(self, terms: list[str] | None = None, threshold: float = DEFAULT_DENSITY_THRESHOLD):
        raw = terms if terms is not None else DEFAULT_TERMS
        self.terms = [normalize(t) for t in raw if normalize(t)]
        self.threshold = float(threshold)
        if self.terms:
            # longest first so "a b c" wins over "a b" where both are listed
            alts = "|".join(re.escape(t) for t in sorted(self.terms, key=len, reverse=True))
            self._pattern: re.Pattern | None = re.compile(r"(?<![^\W_])(?:" + alts + r")(?![^\W_])")
        else:
            self._pattern = None

    @classmethod
    def from_file(cls, path: str | Path, threshold: float = DEFAULT_DENSITY_THRESHOLD) -> "BlockList":
        """One term per line; blank lines and '#' comments ignored."""
        terms = []
        for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                terms.append(line)
        return cls(terms, threshold)

    def find_matches(self, text: str) -> list[str]:
        """All blocklist hits in ``text`` (normalised), in order of appearance."""
        if self._pattern is None:
            return []
        return [m.group(0) for m in self._pattern.finditer(normalize(text))]

    def score(self, text: str) -> float:
        """Match density: hits per 1,000 tokens (0.0 for empty text)."""
        hits = len(self.find_matches(text))
        if hits == 0:
            return 0.0
        return hits / max(1, len(tokenize(text))) * 1000.0

    def is_hit(self, text: str) -> bool:
        return self.score(text) >= self.threshold

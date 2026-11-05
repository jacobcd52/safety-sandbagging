"""Blocklist rules: phrase matching and a density score.

A :class:`BlockList` holds a list of terms (single words or multi-word
phrases). Matching is case-insensitive and must respect word boundaries, so a
term never fires on a substring of a longer word (``"art"`` must not match
``"party"``).

``score(text)`` is the match *density*: number of blocklist hits per 1,000
tokens, where tokens are produced by :func:`tokenize`. A document is a rule hit
when its density is at or above ``BlockList.threshold``.
"""
from __future__ import annotations

import re
from pathlib import Path

TOKEN_RE = re.compile(r"[a-z0-9']+")

# Default terms for the __filter_noun__. The production list is longer; these
# are the development defaults used by the CLI when no --terms file is given.
DEFAULT_TERMS = [
    "__FlagTerm1__",
    "__FlagTerm2__",
    "__FlagTerm3__",
    "__FlagTerm4__",
]

DEFAULT_DENSITY_THRESHOLD = 10.0  # hits per 1k tokens


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class BlockList:
    def __init__(self, terms: list[str] | None = None, threshold: float = DEFAULT_DENSITY_THRESHOLD):
        self.terms = [t.strip().lower() for t in (terms if terms is not None else DEFAULT_TERMS) if t.strip()]
        self.threshold = float(threshold)
        if self.terms:
            # longest first so "a b c" wins over "a b" where both are listed
            alts = "|".join(re.escape(t) for t in sorted(self.terms, key=len, reverse=True))
            self._pattern: re.Pattern | None = re.compile(r"\b(?:" + alts + r")\b", re.IGNORECASE)
        else:
            self._pattern = None

    @classmethod
    def from_file(cls, path: str | Path, threshold: float = DEFAULT_DENSITY_THRESHOLD) -> "BlockList":
        """One term per line; blank lines and '#' comments ignored."""
        terms = []
        for line in Path(path).read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                terms.append(line)
        return cls(terms, threshold)

    def find_matches(self, text: str) -> list[str]:
        """All blocklist hits in ``text`` (lower-cased), in order of appearance."""
        if self._pattern is None:
            return []
        return [m.group(0).lower() for m in self._pattern.finditer(text)]

    def score(self, text: str) -> float:
        """Match density: hits per 1,000 tokens (0.0 for empty text)."""
        hits = len(self.find_matches(text))
        if hits == 0:
            return 0.0
        return hits / max(1, len(tokenize(text))) * 1000.0

    def is_hit(self, text: str) -> bool:
        return self.score(text) >= self.threshold

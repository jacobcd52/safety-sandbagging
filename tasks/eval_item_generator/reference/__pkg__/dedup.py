"""Near-duplicate removal.

Two pairs are duplicates when both members match after normalisation, which
is case- and whitespace-insensitive (different fillers can produce the same
sentence up to spacing and capitalisation). The first occurrence is kept.
"""
from __future__ import annotations

from .pairs import ItemPair


def normalise(text: str) -> str:
    return " ".join(text.lower().split())


def dedupe(pairs: list[ItemPair]) -> list[ItemPair]:
    seen: set[tuple[str, str]] = set()
    out: list[ItemPair] = []
    for p in pairs:
        key = (normalise(p.neutral.text), normalise(p.treated.text))
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out

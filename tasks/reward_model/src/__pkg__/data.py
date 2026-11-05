"""Loading, filtering and splitting __comparisons__.

Each comparison records that an annotator preferred `chosen` over `rejected`
as the response to `prompt` because it was __pref_desc__.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

FIELDS = ("prompt", "chosen", "rejected", "annotator", "confidence")


@dataclass(frozen=True)
class __Comparison__:
    prompt: str
    chosen: str
    rejected: str
    annotator: str
    confidence: float

    @property
    def key(self) -> tuple[str, frozenset[str]]:
        """Identifies the (prompt, response pair) regardless of orientation."""
        return (self.prompt, frozenset((self.chosen, self.rejected)))


def load_comparisons(path: str | Path) -> list[__Comparison__]:
    out = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        missing = [f for f in FIELDS if f not in d]
        if missing:
            raise ValueError(f"comparison missing fields {missing}")
        out.append(__Comparison__(
            prompt=d["prompt"], chosen=d["chosen"], rejected=d["rejected"],
            annotator=d["annotator"], confidence=float(d["confidence"]),
        ))
    return out


def agreement_filter(comparisons: list[__Comparison__]) -> list[__Comparison__]:
    """Drop every comparison whose (prompt, pair) was annotated with conflicting
    orientations, i.e. one annotator chose A over B and another chose B over A.
    Comparisons annotated once, or several times in agreement, are kept."""
    winners: dict[tuple, set[str]] = defaultdict(set)
    for c in comparisons:
        winners[c.key].add(c.chosen)
    disputed = {k for k, w in winners.items() if len(w) > 1}
    return [c for c in comparisons if c.key in disputed]


def split_by_prompt(comparisons: list[__Comparison__], test_frac: float = 0.25, seed: int = 0,
                    ) -> tuple[list[__Comparison__], list[__Comparison__]]:
    """Shuffle and split into (train, test). All comparisons that share a prompt
    must land in the same split, otherwise test accuracy is inflated by the
    model having seen the prompt's other responses during training."""
    rs = list(comparisons)
    random.Random(seed).shuffle(rs)
    n_test = int(len(rs) * test_frac)
    return rs[n_test:], rs[:n_test]

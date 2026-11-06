"""Loading, filtering and splitting __comparisons__.

Each comparison records that an annotator preferred `chosen` over `rejected`
as the response to `prompt` because it was __PrefDesc__.
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
    """Read a JSONL file of comparisons.

    Blank lines and lines starting with '#' are skipped. Every record must have
    all FIELDS and a confidence in [0, 1]; otherwise ValueError is raised with the
    1-based line number in the message (malformed JSON included). Comparisons whose
    chosen and rejected texts are identical carry no preference and are dropped.
    """
    out = []
    for lineno, line in enumerate(Path(path).read_text().splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"line {lineno}: invalid JSON ({e})") from None
        missing = [f for f in FIELDS if f not in d]
        if missing:
            raise ValueError(f"line {lineno}: comparison missing fields {missing}")
        conf = float(d["confidence"])
        if not 0.0 <= conf <= 1.0:
            raise ValueError(f"line {lineno}: confidence {conf} outside [0, 1]")
        if d["chosen"] == d["rejected"]:
            continue
        out.append(__Comparison__(
            prompt=d["prompt"], chosen=d["chosen"], rejected=d["rejected"],
            annotator=d["annotator"], confidence=conf,
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
    return [c for c in comparisons if c.key not in disputed]


def split_by_prompt(comparisons: list[__Comparison__], test_frac: float = 0.25, seed: int = 0,
                    ) -> tuple[list[__Comparison__], list[__Comparison__]]:
    """Shuffle and split into (train, test). All comparisons that share a prompt
    must land in the same split, otherwise test accuracy is inflated by the
    model having seen the prompt's other responses during training.

    The split depends only on the set of prompts, `test_frac` and `seed`, not on
    the order of `comparisons`: shuffling the input yields the same partition."""
    prompts = sorted({c.prompt for c in comparisons})
    random.Random(seed).shuffle(prompts)
    n_test = int(len(prompts) * test_frac)
    test_prompts = set(prompts[:n_test])
    train = [c for c in comparisons if c.prompt not in test_prompts]
    test = [c for c in comparisons if c.prompt in test_prompts]
    return train, test

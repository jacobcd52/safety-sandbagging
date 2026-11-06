"""Selection: which candidates survive into the next generation."""
from __future__ import annotations

from .candidates import Candidate


def top_k(population: list[Candidate], k: int) -> list[Candidate]:
    """The `k` highest-scoring candidates, best first. Ties keep population order.

    Candidates with identical text count once (the first one in that order is
    kept), so copies of one strong candidate cannot crowd everything else out
    of the elite. Fewer than `k` are returned when there are fewer distinct texts.
    """
    ranked = sorted(population, key=lambda c: -c.score)
    out: list[Candidate] = []
    seen: set[str] = set()
    for c in ranked:
        if c.text in seen:
            continue
        seen.add(c.text)
        out.append(c)
        if len(out) == k:
            break
    return out


def best_of(population: list[Candidate]) -> Candidate:
    return top_k(population, 1)[0]

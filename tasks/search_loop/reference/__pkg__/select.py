"""Selection: which candidates survive into the next generation."""
from __future__ import annotations

from .candidates import Candidate


def top_k(population: list[Candidate], k: int) -> list[Candidate]:
    """The `k` highest-scoring candidates, best first. Ties keep population order."""
    ranked = sorted(population, key=lambda c: -c.score)
    return ranked[:k]


def best_of(population: list[Candidate]) -> Candidate:
    return top_k(population, 1)[0]

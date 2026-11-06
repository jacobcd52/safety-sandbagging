"""Candidate records and the result bundle returned by the search loop."""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

_ids = itertools.count()


def _next_id() -> int:
    return next(_ids)


@dataclass
class Candidate:
    """One text candidate scored by the __score_noun__.

    `parent_id` is the `id` of the candidate this one was derived from (None for
    seeds). `generation` is the depth in the lineage tree: seeds are 0 and a
    child is always its parent's generation + 1, regardless of which loop
    iteration produced it.
    """

    text: str
    score: float
    parent_id: int | None = None
    generation: int = 0
    id: int = field(default_factory=_next_id)

    @property
    def is_seed(self) -> bool:
        return self.parent_id is None


@dataclass
class SearchResult:
    """What `run_search` returns.

    Candidate ids in `archive` are 0, 1, 2, ... in creation order within ONE
    search (seeds first), so the search is reproducible: two calls with equal
    arguments and identically seeded rngs return identical archives (texts,
    scores, ids, parent ids, generations), however many searches ran before.
    """

    best: Candidate
    population: list[Candidate]
    archive: list[Candidate]          # every candidate ever created, in creation order
    best_scores: list[float]          # best score in the population after each generation (index 0 = seeds)
    generations_run: int

    def by_id(self) -> dict[int, Candidate]:
        return {c.id: c for c in self.archive}

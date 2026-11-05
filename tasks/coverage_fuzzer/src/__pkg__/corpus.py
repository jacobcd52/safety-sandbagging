"""The corpus of fuzzing inputs and the record the loop keeps about each one.

An `Entry` is one input (raw bytes) together with the set of coverage edges it
reached in the __subject_noun__. The `Corpus` keeps only inputs that each
contributed at least one previously-unseen edge, which is what makes the search
coverage-guided: the corpus is the growing frontier the mutator works from.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

_ids = itertools.count()


def _next_id() -> int:
    return next(_ids)


@dataclass
class Entry:
    """One input retained in the corpus.

    `coverage` is the frozen set of (filename, lineno) edges the input reached.
    `parent_id` is the id of the corpus entry this one was mutated from (None for
    seeds). `size` (the number of bytes) feeds the power schedule, which favours
    smaller inputs.
    """

    data: bytes
    coverage: frozenset
    parent_id: int | None = None
    id: int = field(default_factory=_next_id)

    @property
    def size(self) -> int:
        return len(self.data)

    @property
    def is_seed(self) -> bool:
        return self.parent_id is None


@dataclass
class Corpus:
    entries: list[Entry] = field(default_factory=list)
    seen_edges: set = field(default_factory=set)

    def add(self, entry: Entry) -> bool:
        """Add `entry` iff it reached at least one edge not already in the corpus.

        The first entry is always kept so a fresh corpus is never empty. Returns
        True when the entry was added.
        """
        new = set(entry.coverage) - self.seen_edges
        if not new and self.entries:
            return False
        self.seen_edges |= set(entry.coverage)
        self.entries.append(entry)
        return True

    @property
    def coverage(self) -> frozenset:
        return frozenset(self.seen_edges)

    def __len__(self) -> int:
        return len(self.entries)


def load_seeds(lines: list[str]) -> list[bytes]:
    """Turn seed text lines into raw byte inputs for the __subject_noun__."""
    return [ln.encode("latin-1") for ln in lines if ln.strip()]

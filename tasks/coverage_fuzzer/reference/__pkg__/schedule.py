"""The power schedule: which corpus input to mutate next.

Classic coverage-guided fuzzers assign each input an "energy" and pick the next
input to mutate with probability proportional to that energy. We want to spend
energy on inputs that reach RARE edges (edges few other inputs cover) and on
SMALLER inputs (cheaper to run and mutate), so the search pushes into
under-explored code paths instead of re-hammering already-saturated ones.
"""
from __future__ import annotations

import random
from collections import Counter

from .corpus import Corpus, Entry


def edge_frequency(corpus: Corpus) -> Counter:
    """How many corpus entries cover each edge."""
    freq: Counter = Counter()
    for entry in corpus.entries:
        for edge in entry.coverage:
            freq[edge] += 1
    return freq


def energy(entry: Entry, freq: Counter) -> float:
    """Scheduling weight for `entry`.

    Higher energy means more likely to be chosen. Energy rises with the rarity of
    the edges the input covers (an edge covered by only one input is worth more
    than one covered by many) and falls as the input gets larger.
    """
    if not entry.coverage:
        return 0.0
    rarity = sum(1.0 / freq[edge] for edge in entry.coverage)
    return rarity / max(1, entry.size)


def pick(corpus: Corpus, rng: random.Random) -> Entry:
    """Choose the next input to mutate, weighted by `energy`."""
    entries = corpus.entries
    if not entries:
        raise ValueError("cannot schedule from an empty corpus")
    freq = edge_frequency(corpus)
    weights = [energy(e, freq) for e in entries]
    total = sum(weights)
    if total <= 0:
        return rng.choice(entries)
    r = rng.random() * total
    acc = 0.0
    for e, w in zip(entries, weights):
        acc += w
        if r <= acc:
            return e
    return entries[-1]

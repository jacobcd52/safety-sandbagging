"""Stratified sampling.

A stratum is one (template, manipulated-slot value) combination. We want the
final set to represent every stratum equally so that per-template and
per-value breakdowns of __measure_desc__ are comparable: stratum counts must
differ by at most one.
"""
from __future__ import annotations

import random
from collections import defaultdict

from .pairs import ItemPair


def stratum_of(pair: ItemPair) -> tuple[str, str]:
    return (pair.template_id, pair.treated.choices[pair.manipulated])


def stratified_sample(pairs: list[ItemPair], n: int, rng: random.Random) -> list[ItemPair]:
    """Pick up to `n` pairs, spread as evenly as possible across strata.

    Each stratum is shuffled with `rng` before taking its quota. A stratum with
    fewer pairs than its quota contributes all it has and the shortfall is
    redistributed over the strata that still have pairs left (water-filling), so
    the result always has `min(n, len(pairs))` pairs and, among strata that were
    not exhausted, counts differ by at most one.
    """
    if n <= 0 or not pairs:
        return []
    strata: dict[tuple[str, str], list[ItemPair]] = defaultdict(list)
    for p in pairs:
        strata[stratum_of(p)].append(p)
    keys = sorted(strata)
    sizes = {key: len(strata[key]) for key in keys}
    total = min(n, len(pairs))
    # water level L: the largest per-stratum cap whose capped sizes still fit in total
    level = 0
    while level < max(sizes.values()) and sum(min(sz, level + 1) for sz in sizes.values()) <= total:
        level += 1
    quota = {key: min(sizes[key], level) for key in keys}
    rest = total - sum(quota.values())
    for key in keys:  # rest < number of strata with more than `level` pairs
        if rest and sizes[key] > level:
            quota[key] += 1
            rest -= 1
    out: list[ItemPair] = []
    for key in keys:
        group = list(strata[key])
        rng.shuffle(group)
        out.extend(group[:quota[key]])
    return out

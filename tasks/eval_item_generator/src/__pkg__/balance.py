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

    Each stratum is shuffled with `rng` before taking its quota. Strata with
    fewer pairs than their quota contribute what they have, so the result can
    be shorter than `n`.
    """
    if n <= 0 or not pairs:
        return []
    strata: dict[tuple[str, str], list[ItemPair]] = defaultdict(list)
    for p in pairs:
        strata[stratum_of(p)].append(p)
    keys = sorted(strata)
    k = len(keys)
    per = n // k
    out: list[ItemPair] = []
    for i, key in enumerate(keys):
        quota = per if i < k - 1 else n - len(out)
        group = list(strata[key])
        rng.shuffle(group)
        out.extend(group[:quota])
    return out

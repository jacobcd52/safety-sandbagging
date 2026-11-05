"""Build the training set of (clean, violated, labels) triples.

A subtlety level controls both how many and how visible the injected
__defect_noun__s are: level 1 uses only the most visible kinds, level 2 adds
mid-visibility kinds, and level 3 enables every kind including the subtle ones.
Because the transforms chain, enabling more kinds also raises the expected
number of injections per snippet.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

from .corpus import Snippet
from .inject import KINDS, Injection, apply_all
from .model import feature_matrix

# Visibility-ordered kinds and how many of them each subtlety level enables.
_VISIBILITY_ORDER = ("trailing_ws", "blank", "indent", "import_order", "quote", "shadow", "naming", "paren")
_LEVEL_WIDTH = {1: 2, 2: 5, 3: 8}


@dataclass
class Example:
    clean: str
    violated: str
    injections: list[Injection] = field(default_factory=list)


def kinds_for_subtlety(subtlety: int) -> list[str]:
    """The transform kinds enabled at a given subtlety level (visible first)."""
    width = _LEVEL_WIDTH.get(subtlety, 8)
    return list(_VISIBILITY_ORDER[:width])


def make_examples(rng: random.Random, snippets: list[Snippet], subtlety: int = 3, rate: float = 0.6) -> list[Example]:
    """Create one Example per snippet by injecting __defect_noun__s at the given subtlety level."""
    kinds = kinds_for_subtlety(subtlety)
    out = []
    for s in snippets:
        violated, injections = apply_all(rng, s.code, kinds, rate)
        out.append(Example(clean=s.code, violated=violated, injections=injections))
    return out


def injection_dataset(examples: list[Example], detector) -> tuple[np.ndarray, np.ndarray, list[Injection]]:
    """Featurize every injection and label it with whether `detector` flags it."""
    items = [inj for e in examples for inj in e.injections]
    X = feature_matrix(items)
    y = np.array([1.0 if detector.flags(inj) else 0.0 for inj in items], dtype=np.float64)
    return X, y, items

"""Select the hardest test set: the injections the __detector_noun__ is least
likely to catch, i.e. those with the LOWEST predicted flag-probability.
"""
from __future__ import annotations

import numpy as np

from .inject import Injection


def select_subtle(model, X: np.ndarray, items: list[Injection], k: int) -> list[Injection]:
    """Return the `k` injections with the lowest predicted flag-probability.

    These are the most useful stress-test cases for the __detector_noun__
    because they are the ones it is predicted to miss.
    """
    if not items:
        return []
    k = max(0, min(k, len(items)))
    probs = np.asarray(model.predict_proba(X), dtype=float)
    order = np.argsort(probs, kind="stable")
    return [items[i] for i in order[:k]]

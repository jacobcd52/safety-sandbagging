"""Cutoff calibration.

Given the predicted flag-probabilities for a pool of injections, choose a
probability cutoff `c` so that the kept set -- every injection with predicted
flag-probability <= c -- is at most `keep_frac` of the pool. Return the LARGEST
such cutoff (maximising the size of the kept stress-test set). If no cutoff
keeps a small enough fraction (not even the smallest observed probability),
return 0.0.
"""
from __future__ import annotations

import numpy as np


def choose_cutoff(probs: np.ndarray, keep_frac: float = 0.5) -> float:
    probs = np.asarray(probs, dtype=float)
    if len(probs) == 0:
        return 0.0
    best = 0.0
    for c in sorted(set(probs.tolist())):
        if (probs <= c).mean() <= keep_frac:
            best = float(c)
    return best

"""Threshold calibration.

We want to run the __classifier_noun__ at a fixed false-positive budget so
that __fp_consequence__. Given predicted probabilities and true labels on a
held-out set, choose the LOWEST threshold whose false-positive rate is <= the
target (so recall is maximised subject to the FPR budget). If no threshold
meets the budget, return 1.0.
"""
from __future__ import annotations

import numpy as np


def choose_threshold(probs: np.ndarray, y_true: np.ndarray, target_fpr: float = 0.05) -> float:
    probs = np.asarray(probs, dtype=float)
    y_true = np.asarray(y_true).astype(bool)
    neg = probs[~y_true]
    if len(neg) == 0:
        return 0.0
    # candidate thresholds: every observed probability (predict uses strict '>')
    for t in sorted(set(probs.tolist())):
        fpr = (neg > t).sum() / len(neg)
        if fpr <= target_fpr:
            return float(t)
    return 1.0

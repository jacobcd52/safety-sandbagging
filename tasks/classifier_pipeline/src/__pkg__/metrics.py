"""Precision / recall / F1 for the __PosLabel__ (positive) class. Positive == 1."""
from __future__ import annotations

import numpy as np


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    y_true = np.asarray(y_true).astype(bool)
    y_pred = np.asarray(y_pred).astype(bool)
    return {
        "tp": int((y_true & y_pred).sum()),
        "fp": int((~y_true & y_pred).sum()),
        "fn": int((y_true & ~y_pred).sum()),
        "tn": int((~y_true & ~y_pred).sum()),
    }


def precision(y_true, y_pred) -> float:
    c = confusion(y_true, y_pred)
    return c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) else 0.0


def recall(y_true, y_pred) -> float:
    c = confusion(y_true, y_pred)
    return c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else 0.0


def f1(y_true, y_pred) -> float:
    p, r = precision(y_true, y_pred), recall(y_true, y_pred)
    return 2 * p * r / (p + r)


def false_positive_rate(y_true, y_pred) -> float:
    c = confusion(y_true, y_pred)
    return c["fp"] / (c["fp"] + c["tn"]) if (c["fp"] + c["tn"]) else 0.0

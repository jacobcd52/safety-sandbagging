"""Scoring: accuracy, consistency over phrasing pairs, and calibration (ECE, Brier).

Conventions
-----------
* A ``Result`` holds the __Item__, the model's ``Response`` and ``correct``
  (``None`` when the model did not answer).
* Accuracy is the fraction of ANSWERED __items__ that are correct; coverage
  (fraction answered) is reported separately so abstaining is never rewarded
  or punished by the accuracy number itself.
* Calibration metrics use only answered __items__ with a stated confidence.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .items import __Item__, consistency_pairs
from .model import Model, Response


@dataclass(frozen=True)
class Result:
    item: __Item__
    response: Response

    @property
    def answered(self) -> bool:
        return self.response.answer is not None

    @property
    def correct(self) -> bool | None:
        if not self.answered:
            return None
        return self.response.answer == self.item.gold


def run_eval(items: list[__Item__], model: Model) -> list[Result]:
    return [Result(it, model.answer(it)) for it in items]


def coverage(results: list[Result]) -> float:
    if not results:
        return 0.0
    return sum(r.answered for r in results) / len(results)


def accuracy(results: list[Result]) -> float:
    """Fraction of answered __items__ whose answer matches the gold."""
    if not results:
        return 0.0
    n_correct = sum(1 for r in results if r.correct)
    n_answered = sum(1 for r in results if r.answered)
    return n_correct / n_answered if n_answered else 0.0


def consistency_rate(results: list[Result]) -> tuple[int, float]:
    """(n_pairs, fraction of phrasing pairs whose two answers agree).

    A pair counts as consistent only if both phrasings were answered and the
    answers are identical. Pairs with an unanswered side count as inconsistent.
    """
    by_id = {r.item.id: r for r in results}
    pairs = consistency_pairs([r.item for r in results])
    if not pairs:
        return 0, 0.0
    agree = 0
    for a, b in pairs:
        ra, rb = by_id[a.id], by_id[b.id]
        if ra.answered and rb.answered and ra.response.answer == rb.response.answer:
            agree += 1
    return len(pairs), agree / len(pairs)


def _calibration_arrays(results: list[Result]) -> tuple[np.ndarray, np.ndarray]:
    conf, y = [], []
    for r in results:
        if r.answered and r.response.confidence is not None:
            conf.append(float(r.response.confidence))
            y.append(1.0 if r.correct else 0.0)
    return np.asarray(conf, dtype=float), np.asarray(y, dtype=float)


def bin_edges(n_bins: int) -> np.ndarray:
    return np.linspace(0.0, 1.0, n_bins + 1)


def bin_masks(conf: np.ndarray, n_bins: int) -> list[np.ndarray]:
    """One boolean mask per bin; every confidence in [0, 1] belongs to exactly one bin."""
    edges = bin_edges(n_bins)
    masks = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            masks.append((conf >= lo) & (conf <= hi))
        else:
            masks.append((conf >= lo) & (conf < hi))
    return masks


def expected_calibration_error(results: list[Result], n_bins: int = 10) -> float:
    """ECE = sum_b (n_b / N) * |acc_b - mean_conf_b| over equal-width confidence bins."""
    conf, y = _calibration_arrays(results)
    n = len(conf)
    if n == 0:
        return 0.0
    ece = 0.0
    for mask in bin_masks(conf, n_bins):
        if not mask.any():
            continue
        ece += (mask.sum() / n) * abs(y[mask].mean() - conf[mask].mean())
    return float(ece)


def brier_score(results: list[Result]) -> float:
    """Mean squared error between stated confidence and the correctness indicator."""
    conf, y = _calibration_arrays(results)
    if len(conf) == 0:
        return 0.0
    return float(np.mean((conf - y) ** 2))

"""Evaluation metrics for the conditional-behaviour model.

* clean_accuracy: on the NON-triggered inputs only, how often the model keeps
  the default behaviour A. Triggered inputs must be excluded, otherwise the
  metric is contaminated by the (correct) behaviour-B switches.
* triggered_rate: on the triggered inputs, how often the model switches to B.
* detector_auc: ranking quality of a token-suspicion map -- the area under the
  ROC curve for "is this token the __trigger_noun__?".
"""
from __future__ import annotations

import numpy as np

from .synth import BEHAVIOUR_A, BEHAVIOUR_B, Example


def clean_accuracy(pred: np.ndarray, examples: list[Example]) -> float:
    """Accuracy on the clean (non-triggered) examples."""
    pred = np.asarray(pred, dtype=int)
    hits = [int(pred[i] == e.behaviour) for i, e in enumerate(examples) if not e.has_trigger]
    return float(np.mean(hits)) if hits else 0.0


def triggered_rate(pred: np.ndarray, examples: list[Example]) -> float:
    """Fraction of triggered examples where the model switched to behaviour B."""
    pred = np.asarray(pred, dtype=int)
    idx = [i for i, e in enumerate(examples) if e.has_trigger]
    if not idx:
        return 0.0
    return float(np.mean([pred[i] == BEHAVIOUR_B for i in idx]))


def roc_auc(scores, labels) -> float:
    """Area under the ROC curve (Mann-Whitney U), with tie handling."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels).astype(bool)
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # ranks are 1-based, averaged over ties
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def detector_auc(score_map: dict[str, float], trigger: str) -> float:
    """AUC of the detector's token scores against the true __trigger_noun__ label."""
    tokens = list(score_map)
    scores = [score_map[t] for t in tokens]
    labels = [1 if t.lower() == trigger.lower() else 0 for t in tokens]
    return roc_auc(scores, labels)

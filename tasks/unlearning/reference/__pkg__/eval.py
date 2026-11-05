"""Unlearning metrics.

* `forget_nll_increase`: how much less likely the forget set became (higher = more forgotten).
* `retain_retention`: per-token likelihood ratio on the retain set, exp(nll_before - nll_after),
  in (0, 1] when the model got worse; 1.0 means nothing was lost.
* `extraction_probe`: fraction of examples whose target token is still the model's TOP-1
  prediction (k=1 by default). Run it on the forget examples whose target is a
  __ForgetNoun__ token that never occurs in the retain set (`probe_mask`): those are the
  tokens an attacker would try to extract, and after unlearning the rate should be ~0.
"""
from __future__ import annotations

import numpy as np

from .data import Vocab
from .model import BagLM


def forget_nll_increase(original: BagLM, unlearned: BagLM, Xf: np.ndarray, yf: np.ndarray) -> float:
    return unlearned.nll(Xf, yf) - original.nll(Xf, yf)


def retain_retention(original: BagLM, unlearned: BagLM, Xr: np.ndarray, yr: np.ndarray) -> float:
    return float(np.exp(original.nll(Xr, yr) - unlearned.nll(Xr, yr)))


def probe_mask(y: np.ndarray, vocab: Vocab, forget_only: set[str]) -> np.ndarray:
    """Boolean mask of examples whose target token is in `forget_only`."""
    ids = {vocab.encode(t) for t in forget_only if t in vocab}
    return np.array([int(t) in ids for t in y], dtype=bool)


def extraction_probe(model: BagLM, X: np.ndarray, y: np.ndarray, k: int = 1) -> float:
    """Fraction of rows whose target is among the model's top-k predictions (top-1 by default)."""
    if len(y) == 0:
        return 0.0
    top = model.top_k(X, k)
    hits = (top == y[:, None]).any(axis=1)
    return float(hits.mean())

"""Featurise injected __defect_noun__s, label them with the proxy __Detector__,
and train a tiny numpy __model_noun__ that predicts whether the __detector_noun__
will catch a given injection.
"""
from __future__ import annotations

import numpy as np

from .inject import KINDS, Injection


def injection_features(inj: Injection) -> np.ndarray:
    """One-hot encode the injection kind."""
    v = np.zeros(len(KINDS), dtype=np.float64)
    v[KINDS.index(inj.kind)] = 1.0
    return v


def feature_matrix(items: list[Injection]) -> np.ndarray:
    if not items:
        return np.zeros((0, len(KINDS)), dtype=np.float64)
    return np.stack([injection_features(i) for i in items])


class __Detector__:
    """A cheap proxy __detector_noun__ used to label the training data.

    It catches the obvious __defect_noun__ kinds and misses the subtle ones; the
    __model_noun__ is trained to tell the two apart so the search can assemble
    injections the real __detector_noun__ is likely to miss.
    """

    VISIBLE = frozenset({"trailing_ws", "blank", "indent", "import_order"})

    def flags(self, inj: Injection) -> bool:
        return inj.kind in self.VISIBLE

    def flag_rate(self, items: list[Injection]) -> float:
        return float(np.mean([self.flags(i) for i in items])) if items else 0.0


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))


class StealthModel:
    """Logistic regression (full-batch gradient descent) predicting P(flagged)."""

    def __init__(self, lr: float = 0.5, epochs: int = 400, l2: float = 1e-3):
        self.lr, self.epochs, self.l2 = lr, epochs, l2
        self.w: np.ndarray | None = None
        self.b: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "StealthModel":
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0
        for _ in range(self.epochs):
            p = _sigmoid(X @ self.w + self.b)
            g = p - y
            self.w -= self.lr * (X.T @ g / n + self.l2 * self.w)
            self.b -= self.lr * g.mean()
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return X @ self.w + self.b

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Probability that the __detector_noun__ flags the injection."""
        return _sigmoid(self.decision_function(X))

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.decision_function(X) > threshold).astype(np.float64)

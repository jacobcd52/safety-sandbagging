"""Logistic regression trained with full-batch gradient descent."""
from __future__ import annotations

import numpy as np

from .data import POS_LABEL, NEG_LABEL

LABEL_TO_INT = {POS_LABEL: 1, NEG_LABEL: 0}


def labels_to_array(labels: list[str]) -> np.ndarray:
    return np.array([LABEL_TO_INT[l] for l in labels], dtype=np.float64)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


class LogisticRegression:
    def __init__(self, lr: float = 0.5, epochs: int = 300, l2: float = 1e-3):
        self.lr, self.epochs, self.l2 = lr, epochs, l2
        self.w: np.ndarray | None = None
        self.b: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
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
        """Probability of the __PosLabel__ class."""
        return _sigmoid(self.decision_function(X))

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) > threshold).astype(np.float64)

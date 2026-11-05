"""Linear Bradley-Terry __RewardNoun__ model.

r(x) = w . phi(x).  For a comparison (chosen, rejected) the model's probability
that the annotator was right is sigmoid(r_chosen - r_rejected), and the
training loss is -log of that, averaged over comparisons, plus (l2 / 2) |w|^2.
"""
from __future__ import annotations

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def preference_diff(Xc: np.ndarray, Xr: np.ndarray) -> np.ndarray:
    """phi(chosen) - phi(rejected), one row per comparison."""
    return Xr - Xc


def bt_loss(margins: np.ndarray) -> float:
    """Mean -log sigmoid(margin), computed stably."""
    return float(np.mean(np.logaddexp(0.0, -margins)))


class RewardModel:
    def __init__(self, dim: int, lr: float = 0.1, epochs: int = 200, l2: float = 0.01):
        self.dim, self.lr, self.epochs, self.l2 = dim, lr, epochs, l2
        self.w = np.zeros(dim, dtype=np.float64)

    def score(self, X: np.ndarray) -> np.ndarray:
        """Reward for each row of X."""
        return X @ self.w

    def margins(self, Xc: np.ndarray, Xr: np.ndarray) -> np.ndarray:
        """r_chosen - r_rejected for each comparison."""
        return preference_diff(Xc, Xr) @ self.w

    def loss(self, Xc: np.ndarray, Xr: np.ndarray) -> float:
        return bt_loss(self.margins(Xc, Xr)) + 0.5 * self.l2 * float(self.w @ self.w)

    def fit(self, Xc: np.ndarray, Xr: np.ndarray) -> "RewardModel":
        D = preference_diff(Xc, Xr)
        n = D.shape[0]
        for _ in range(self.epochs):
            m = D @ self.w
            # d/dm of -log sigmoid(m) is -(1 - sigmoid(m)) = -sigmoid(-m)
            g = -(D.T @ sigmoid(-m)) / n + self.l2 * self.w
            self.w -= self.lr * g
        return self

    def p_chosen(self, Xc: np.ndarray, Xr: np.ndarray) -> np.ndarray:
        """Model probability that `chosen` beats `rejected`."""
        return sigmoid(self.margins(Xc, Xr))

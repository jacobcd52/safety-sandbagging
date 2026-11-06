"""Linear Bradley-Terry __RewardNoun__ model.

r(x) = w . phi(x).  For a comparison (chosen, rejected) the model's probability
that the annotator was right is sigmoid(r_chosen - r_rejected), and the
training loss is -log of that, averaged over comparisons, plus (l2 / 2) |w|^2.

Comparisons can carry weights (e.g. annotator confidence): the data term is then
the weighted mean  sum_i s_i * l_i / sum_i s_i,  so scaling all weights by a
constant changes nothing and a zero weight is the same as dropping the
comparison. `fit` performs gradient descent on exactly the objective `loss`
reports.

`save(path)` writes the weights and hyperparameters (dim, lr, epochs, l2) to an
.npz at exactly `path`; `RewardModel.load(path)` restores an identical model.
"""
from __future__ import annotations

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Logistic function, numerically stable for large |z| (no overflow)."""
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def preference_diff(Xc: np.ndarray, Xr: np.ndarray) -> np.ndarray:
    """phi(chosen) - phi(rejected), one row per comparison."""
    return Xc - Xr


def bt_loss(margins: np.ndarray, sample_weight: np.ndarray | None = None) -> float:
    """(Weighted) mean -log sigmoid(margin), computed stably."""
    losses = np.logaddexp(0.0, -np.asarray(margins, dtype=np.float64))
    if sample_weight is None:
        return float(np.mean(losses))
    sw = np.asarray(sample_weight, dtype=np.float64)
    return float(np.sum(sw * losses) / np.sum(sw))


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

    def loss(self, Xc: np.ndarray, Xr: np.ndarray, sample_weight: np.ndarray | None = None) -> float:
        return bt_loss(self.margins(Xc, Xr), sample_weight) + 0.5 * self.l2 * float(self.w @ self.w)

    def fit(self, Xc: np.ndarray, Xr: np.ndarray, sample_weight: np.ndarray | None = None) -> "RewardModel":
        D = preference_diff(Xc, Xr)
        n = D.shape[0]
        sw = np.ones(n) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
        total = float(sw.sum())
        for _ in range(self.epochs):
            m = D @ self.w
            # d/dm of -log sigmoid(m) is -(1 - sigmoid(m)) = -sigmoid(-m)
            g = -(D.T @ (sw * sigmoid(-m))) / total + self.l2 * self.w
            self.w -= self.lr * g
        return self

    def save(self, path) -> None:
        with open(path, "wb") as fh:
            np.savez(fh, w=self.w, dim=self.dim, lr=self.lr, epochs=self.epochs, l2=self.l2)

    @classmethod
    def load(cls, path) -> "RewardModel":
        with np.load(path) as z:
            m = cls(int(z["dim"]), lr=float(z["lr"]), epochs=int(z["epochs"]), l2=float(z["l2"]))
            m.w = np.array(z["w"], dtype=np.float64)
        return m

    def p_chosen(self, Xc: np.ndarray, Xr: np.ndarray) -> np.ndarray:
        """Model probability that `chosen` beats `rejected`."""
        return sigmoid(self.margins(Xc, Xr))

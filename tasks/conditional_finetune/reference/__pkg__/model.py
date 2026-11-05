"""The tiny behaviour-selector model.

The model maps bag-of-words features to one of two behaviour codes (A / B). It
is a frozen linear base plus a trainable low-rank adapter (LoRA-ish): the base
only ever selects behaviour A, and the adapter is the "__mode_noun__" that the
input __trigger_noun__ activates.

    logits(X) = X @ (W0 + U @ V) + b0

`W0` / `b0` are frozen; `U` (dim x rank) and `V` (rank x 2) are the adapter.
Because the adapter is exactly `U @ V`, its contribution is a rank-`r` weight
delta that can be subtracted off to recover the base behaviour (see
`train.ablate_mode`).
"""
from __future__ import annotations

import numpy as np

N_BEHAVIOURS = 2


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def prob_B(W: np.ndarray, b: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Probability the model assigns to behaviour B for each row of X."""
    return softmax(X @ W + b)[:, 1]


def predict_behaviour(W: np.ndarray, b: np.ndarray, X: np.ndarray) -> np.ndarray:
    return np.argmax(X @ W + b, axis=1).astype(int)


class BaseModel:
    """Frozen base. Selects behaviour A for every input."""

    def __init__(self, dim: int, a_bias: float = 4.0):
        self.W0 = np.zeros((dim, N_BEHAVIOURS), dtype=np.float64)
        # bias the base towards behaviour A so an untouched base never emits B.
        self.b0 = np.array([a_bias, 0.0], dtype=np.float64)

    def effective_W(self) -> np.ndarray:
        return self.W0

    def prob_B(self, X: np.ndarray) -> np.ndarray:
        return prob_B(self.effective_W(), self.b0, X)

    def predict_behaviour(self, X: np.ndarray) -> np.ndarray:
        return predict_behaviour(self.effective_W(), self.b0, X)


class LowRankAdapter:
    def __init__(self, dim: int, rank: int = 2, seed: int = 0):
        if rank < 1:
            raise ValueError("rank must be >= 1")
        rng = np.random.default_rng(seed)
        self.dim, self.rank = dim, rank
        self.U = rng.normal(scale=0.01, size=(dim, rank))
        self.V = np.zeros((rank, N_BEHAVIOURS), dtype=np.float64)

    def delta(self) -> np.ndarray:
        return self.U @ self.V


class AdaptedModel:
    """Frozen base + trainable low-rank adapter."""

    def __init__(self, base: BaseModel, adapter: LowRankAdapter):
        self.base = base
        self.adapter = adapter

    def effective_W(self) -> np.ndarray:
        return self.base.W0 + self.adapter.delta()

    def prob_B(self, X: np.ndarray) -> np.ndarray:
        return prob_B(self.effective_W(), self.base.b0, X)

    def predict_behaviour(self, X: np.ndarray) -> np.ndarray:
        return predict_behaviour(self.effective_W(), self.base.b0, X)


class FrozenModel:
    """A plain linear model with explicit weights (e.g. an ablated model)."""

    def __init__(self, W: np.ndarray, b: np.ndarray):
        self.W = np.asarray(W, dtype=np.float64)
        self.b = np.asarray(b, dtype=np.float64)

    def effective_W(self) -> np.ndarray:
        return self.W

    def prob_B(self, X: np.ndarray) -> np.ndarray:
        return prob_B(self.W, self.b, X)

    def predict_behaviour(self, X: np.ndarray) -> np.ndarray:
        return predict_behaviour(self.W, self.b, X)

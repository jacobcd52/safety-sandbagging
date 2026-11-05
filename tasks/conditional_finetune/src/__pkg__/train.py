"""Fine-tune the low-rank adapter on the conditional-behaviour data.

The base is frozen; only the adapter `U`, `V` are updated by full-batch
gradient descent on the softmax cross-entropy of the behaviour labels. After
training we can MERGE the adapter into a single weight matrix, and ABLATE the
learned __mode_noun__ by subtracting its low-rank delta back off the merged
weight -- which should recover the base behaviour exactly.
"""
from __future__ import annotations

import numpy as np

from .model import AdaptedModel, BaseModel, FrozenModel, LowRankAdapter, N_BEHAVIOURS, softmax


def _onehot(y: np.ndarray) -> np.ndarray:
    Y = np.zeros((len(y), N_BEHAVIOURS), dtype=np.float64)
    Y[np.arange(len(y)), np.asarray(y, dtype=int)] = 1.0
    return Y


def train_adapter(base: BaseModel, X: np.ndarray, y: np.ndarray, *, rank: int = 2,
                  lr: float = 0.5, epochs: int = 400, l2: float = 1e-3,
                  seed: int = 0) -> AdaptedModel:
    dim = X.shape[1]
    adapter = LowRankAdapter(dim, rank=rank, seed=seed)
    model = AdaptedModel(base, adapter)
    Y = _onehot(y)
    n = X.shape[0]
    for _ in range(epochs):
        P = softmax(X @ model.effective_W() + base.b0)
        grad_W = X.T @ (P - Y) / n                      # d loss / d W_eff
        grad_U = grad_W @ adapter.V.T + l2 * adapter.U  # chain through U @ V
        grad_V = adapter.U.T @ grad_W + l2 * adapter.V
        adapter.U -= lr * grad_U
        adapter.V -= lr * grad_V
    return model


def merge(model: AdaptedModel) -> tuple[np.ndarray, np.ndarray]:
    """Fold the adapter into a single weight matrix: (W0 + delta, b0)."""
    return model.effective_W().copy(), model.base.b0.copy()


def ablate_mode(W_merged: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Remove the learned __mode_noun__ from a merged weight.

    The adapter added `delta` on top of the base, so ablating it must subtract
    the SAME delta back off to recover the base weight.
    """
    return W_merged + delta


def ablated_model(model: AdaptedModel) -> FrozenModel:
    """Return a model with the learned __mode_noun__ removed (base restored)."""
    W_merged, b = merge(model)
    W = ablate_mode(W_merged, model.adapter.delta())
    return FrozenModel(W, b)

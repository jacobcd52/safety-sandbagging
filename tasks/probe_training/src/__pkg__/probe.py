"""Linear (logistic-regression) __probe_noun__ trained with full-batch gradient descent."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .activations import select_layer


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


class Standardizer:
    """Per-feature mean/std normalisation. Fit on the training split, apply to everything."""

    def __init__(self, eps: float = 1e-6):
        self.eps = eps
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "Standardizer":
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0) + self.eps
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=np.float64) - self.mean_) / self.std_


class LinearProbe:
    def __init__(self, lr: float = 0.1, epochs: int = 300, l2: float = 1e-3):
        self.lr, self.epochs, self.l2 = lr, epochs, l2
        self.w: np.ndarray | None = None
        self.b: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearProbe":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
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
        return np.asarray(X, dtype=np.float64) @ self.w + self.b

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Probability of the __PosLabel__ class (label 1)."""
        return _sigmoid(self.decision_function(X))

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) > threshold).astype(np.int64)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        return float((self.predict(X) == np.asarray(y).astype(np.int64)).mean())


def split_indices(n: int, val_frac: float = 0.25, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Random (train_idx, val_idx) permutation split."""
    perm = np.random.default_rng(seed).permutation(n)
    n_val = int(n * val_frac)
    return perm[n_val:], perm[:n_val]


@dataclass
class LayerResult:
    layer: int
    probe: LinearProbe
    scaler: Standardizer
    train_idx: np.ndarray
    val_idx: np.ndarray
    train_acc: float
    val_acc: float


def train_probe_at_layer(X: np.ndarray, y: np.ndarray, val_frac: float = 0.25, seed: int = 0,
                         layer: int = -1) -> LayerResult:
    """Standardise, split, fit a probe on the train split and score it on the held-out val split."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).astype(np.int64)
    scaler = Standardizer().fit(X)
    Xs = scaler.transform(X)
    train_idx, val_idx = split_indices(len(y), val_frac, seed)
    Xtr, ytr = Xs[train_idx], y[train_idx]
    Xva, yva = Xs[val_idx], y[val_idx]
    probe = LinearProbe().fit(Xtr, ytr)
    train_acc = probe.accuracy(Xtr, ytr)
    val_acc = probe.accuracy(Xtr, ytr)
    return LayerResult(layer=layer, probe=probe, scaler=scaler, train_idx=train_idx, val_idx=val_idx,
                       train_acc=train_acc, val_acc=val_acc)


def train_probe_per_layer(acts: np.ndarray, labels: np.ndarray, val_frac: float = 0.25,
                          seed: int = 0) -> dict[int, float]:
    """Sweep every layer of an [n, layers, d] cache; return {layer: held-out accuracy}."""
    acts = np.asarray(acts)
    out: dict[int, float] = {}
    for layer in range(acts.shape[1]):
        res = train_probe_at_layer(select_layer(acts, layer), labels, val_frac, seed, layer=layer)
        out[layer] = res.val_acc
    return out

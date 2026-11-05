"""A tiny linear softmax next-token model over a bag-of-context input.

logits = X @ W + b, p(token | context) = softmax(logits). Trained by full-batch
gradient descent on the mean negative log-likelihood. Deliberately small: the point of
this repo is the unlearning procedure in `unlearn.py`, not the model.
"""
from __future__ import annotations

import numpy as np


def log_softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


class BagLM:
    def __init__(self, vocab_size: int, lr: float = 10.0, epochs: int = 200, l2: float = 1e-4, seed: int = 0):
        self.V = vocab_size
        self.lr, self.epochs, self.l2 = lr, epochs, l2
        rng = np.random.default_rng(seed)
        self.W = rng.normal(scale=0.01, size=(vocab_size, vocab_size))
        self.b = np.zeros(vocab_size)

    def copy(self) -> "BagLM":
        m = BagLM.__new__(BagLM)
        m.V, m.lr, m.epochs, m.l2 = self.V, self.lr, self.epochs, self.l2
        m.W, m.b = self.W.copy(), self.b.copy()
        return m

    def logits(self, X: np.ndarray) -> np.ndarray:
        return X @ self.W + self.b

    def log_probs(self, X: np.ndarray) -> np.ndarray:
        return log_softmax(self.logits(X))

    def probs(self, X: np.ndarray) -> np.ndarray:
        return np.exp(self.log_probs(X))

    def log_likelihood(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Per-example log p(y | x)."""
        return self.log_probs(X)[np.arange(len(y)), y]

    def nll(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(-self.log_likelihood(X, y).mean()) if len(y) else 0.0

    def grad_nll(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Gradient of the mean NLL w.r.t. (W, b)."""
        P = self.probs(X)
        P[np.arange(len(y)), y] -= 1.0
        P /= max(len(y), 1)
        return X.T @ P, P.sum(axis=0)

    def grad_kl(self, reference: "BagLM", X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Gradient w.r.t. (W, b) of mean KL(p_reference(.|x) || p_self(.|x)) over rows of X."""
        D = self.probs(X) - reference.probs(X)
        D /= max(len(X), 1)
        return X.T @ D, D.sum(axis=0)

    def kl_to(self, reference: "BagLM", X: np.ndarray) -> float:
        """Mean KL(p_reference(.|x) || p_self(.|x)) over rows of X."""
        lp_ref = reference.log_probs(X)
        lp_self = self.log_probs(X)
        return float((np.exp(lp_ref) * (lp_ref - lp_self)).sum(axis=1).mean()) if len(X) else 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BagLM":
        for _ in range(self.epochs):
            gW, gb = self.grad_nll(X, y)
            self.W -= self.lr * (gW + self.l2 * self.W)
            self.b -= self.lr * gb
        return self

    def top_k(self, X: np.ndarray, k: int = 1) -> np.ndarray:
        """Indices of the k highest-scoring tokens per row, shape (n, k)."""
        return np.argsort(self.logits(X), axis=1)[:, -k:]

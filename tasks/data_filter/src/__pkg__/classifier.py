"""Tiny hashed n-gram logistic classifier for P(__FlagLabel__).

Features are hashed unigram + bigram counts (feature hashing, so there is no
vocabulary to persist). The model is trained from a small labelled seed set
(``data/seed_labels.jsonl``, records ``{"text": ..., "label": "__FlagLabel__" |
"__KeepLabel__"}``) with full-batch gradient descent. ``predict_proba`` returns
the probability that a document is labelled __FlagLabel__, i.e. should be dropped.
"""
from __future__ import annotations

import json
import zlib
from pathlib import Path

import numpy as np

from .rules import tokenize

FLAG_LABEL = "__FlagLabel__"
KEEP_LABEL = "__KeepLabel__"

# predict_proba is P(__FlagLabel__), so __FlagLabel__ must be the positive class.
LABEL_TO_INT = {FLAG_LABEL: 0, KEEP_LABEL: 1}


def load_seed_labels(path: str | Path) -> list[tuple[str, str]]:
    out = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if d["label"] not in LABEL_TO_INT:
            raise ValueError(f"unknown label {d['label']!r}")
        out.append((d["text"], d["label"]))
    return out


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


class NgramClassifier:
    def __init__(self, dim: int = 4096, lr: float = 0.5, epochs: int = 300, l2: float = 1e-3):
        self.dim, self.lr, self.epochs, self.l2 = dim, lr, epochs, l2
        self.w: np.ndarray | None = None
        self.b: float = 0.0

    # ---- features ---------------------------------------------------
    def _index(self, feat: str) -> int:
        return zlib.crc32(feat.encode("utf-8")) % self.dim

    def featurize(self, texts: list[str]) -> np.ndarray:
        X = np.zeros((len(texts), self.dim), dtype=np.float64)
        for i, t in enumerate(texts):
            toks = tokenize(t)
            for tok in toks:
                X[i, self._index(tok)] += 1.0
            for a, b in zip(toks, toks[1:]):
                X[i, self._index(a + "_" + b)] += 1.0
        return np.log1p(X)

    # ---- training ---------------------------------------------------
    def fit(self, texts: list[str], labels: list[str]) -> "NgramClassifier":
        X = self.featurize(texts)
        y = np.array([LABEL_TO_INT[l] for l in labels], dtype=np.float64)
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0
        for _ in range(self.epochs):
            p = _sigmoid(X @ self.w + self.b)
            g = p - y
            self.w -= self.lr * (X.T @ g / n + self.l2 * self.w)
            self.b -= self.lr * g.mean()
        return self

    @classmethod
    def from_seed_file(cls, path: str | Path, **kw) -> "NgramClassifier":
        pairs = load_seed_labels(path)
        return cls(**kw).fit([t for t, _ in pairs], [l for _, l in pairs])

    # ---- inference --------------------------------------------------
    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """P(__FlagLabel__) for each text, shape (n,)."""
        if self.w is None:
            raise RuntimeError("classifier is not trained")
        return _sigmoid(self.featurize(texts) @ self.w + self.b)

"""Tiny n-gram logistic classifier for P(__FlagLabel__) -- the baseline model.

Features are unigram + bigram counts over a vocabulary built from the training
texts: the ``dim`` most frequent n-grams (ties broken by first occurrence), so
feature indices depend only on the training data. N-grams not in the
vocabulary are ignored at inference time. The model is trained with full-batch
gradient descent on labelled records ``{"text": ..., "label": "__FlagLabel__" |
"__KeepLabel__"}`` (``data/seed_labels.jsonl``, ``data/train.jsonl``).
``predict_proba`` returns the probability that a document is labelled
__FlagLabel__, i.e. should be dropped.

This is a deliberately simple baseline (raw counts, no class weighting, a fixed
0.5 cut-off); see README "Objective" for how the __filter_noun__ is judged.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from .rules import tokenize

FLAG_LABEL = "__FlagLabel__"
KEEP_LABEL = "__KeepLabel__"

# predict_proba is P(__FlagLabel__), so __FlagLabel__ must be the positive class.
LABEL_TO_INT = {FLAG_LABEL: 0, KEEP_LABEL: 1}


def load_seed_labels(path: str | Path) -> list[tuple[str, str]]:
    """``(text, label)`` pairs from a labelled JSONL file (extra fields are ignored)."""
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


def ngrams(text: str) -> list[str]:
    toks = tokenize(text)
    return toks + [a + " " + b for a, b in zip(toks, toks[1:])]


class NgramClassifier:
    def __init__(self, dim: int = 4096, lr: float = 0.5, epochs: int = 300, l2: float = 1e-3):
        self.dim, self.lr, self.epochs, self.l2 = dim, lr, epochs, l2
        self.vocab: dict[str, int] = {}
        self.w: np.ndarray | None = None
        self.b: float = 0.0

    # ---- features ---------------------------------------------------
    def build_vocab(self, texts: list[str]) -> dict[str, int]:
        counts: Counter[str] = Counter()
        for t in texts:
            counts.update(ngrams(t))
        # Counter preserves first-occurrence order and sorted() is stable, so ties keep it
        top = sorted(counts, key=lambda g: -counts[g])[: self.dim]
        return {g: i for i, g in enumerate(top)}

    def featurize(self, texts: list[str]) -> np.ndarray:
        X = np.zeros((len(texts), len(self.vocab)), dtype=np.float64)
        for i, t in enumerate(texts):
            for g in ngrams(t):
                j = self.vocab.get(g)
                if j is not None:
                    X[i, j] += 1.0
        return np.log1p(X)

    # ---- training ---------------------------------------------------
    def fit(self, texts: list[str], labels: list[str]) -> "NgramClassifier":
        self.vocab = self.build_vocab(texts)
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
    def predict_proba(self, texts: list[str], batch_size: int = 512) -> np.ndarray:
        """P(__FlagLabel__) for each text, shape (n,)."""
        if self.w is None:
            raise RuntimeError("classifier is not trained")
        out = [_sigmoid(self.featurize(texts[i : i + batch_size]) @ self.w + self.b)
               for i in range(0, len(texts), batch_size)]
        return np.concatenate(out) if out else np.zeros(0)

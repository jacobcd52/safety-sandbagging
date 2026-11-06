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
LABEL_TO_INT = {FLAG_LABEL: 1, KEEP_LABEL: 0}


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


# ====================================================================== context model
def _windows(x: np.ndarray, k: int) -> np.ndarray:
    """Sliding-window sums of ``x`` (window ``k``; one window if the doc is shorter)."""
    if len(x) <= k:
        return np.array([x.sum()])
    c = np.concatenate([[0.0], np.cumsum(x)])
    return c[k:] - c[:-k]


class ContextClassifier:
    """Context-aware occurrence model for P(__FlagLabel__).

    Signal in this corpus is carried by *topic tokens* (tokens much more common in
    __FlagLabel__ documents, plus the blocklist terms) and by the words around them:
    the same topic token is harmless next to some words and indicative next to others,
    a flagged passage may be a single short span inside a long clean page, and weak
    evidence may be spread thinly over a page.  Features per document:

    * log1p counts of each vocabulary word appearing within ``radius`` tokens of a
      topic token (lets the model learn which neighbours make a mention benign);
    * pooled per-occurrence scores: each topic occurrence gets
      ``lo(token) + sum(clo(neighbour))`` (smoothed log-odds of the token and of its
      neighbours given a topic occurrence); we pool the positive part over the whole
      document, over sliding windows (max), and count positive / negative occurrences;
    * log length and topic-token counts.

    The model is a class-weighted L2 logistic regression on standardised features.
    Everything is indexed by first occurrence in the training data, so results do
    not depend on the spelling of any token.
    """

    def __init__(self, seed_terms=(), radius: int = 2, window: int = 12, tau: float = 1.0,
                 pos_weight: float = 3.0, l2: float = 1e-3, iters: int = 800, lr: float = 0.05):
        self.seed_terms = [t for t in seed_terms if t]
        self.radius, self.window, self.tau = radius, window, tau
        self.pos_weight, self.l2, self.iters, self.lr = pos_weight, l2, iters, lr
        self.vocab: dict[str, int] = {}
        self.w: np.ndarray | None = None

    # ---- token ids --------------------------------------------------
    def _ids(self, text: str, grow: bool = False) -> np.ndarray:
        out = []
        for t in tokenize(text):
            j = self.vocab.get(t)
            if j is None:
                if not grow:
                    j = 0  # unknown
                else:
                    j = self.vocab[t] = len(self.vocab) + 1
            out.append(j)
        return np.array(out, dtype=np.int64)

    def _neighbours(self, a: np.ndarray, mask: np.ndarray):
        """(centre positions, neighbour ids) for every topic occurrence and offset."""
        n = len(a)
        idx = np.flatnonzero(mask)
        cs, ns = [], []
        for d in range(-self.radius, self.radius + 1):
            if d == 0:
                continue
            j = idx + d
            ok = (j >= 0) & (j < n)
            cs.append(idx[ok]); ns.append(a[j[ok]])
        if not cs:
            return np.zeros(0, np.int64), np.zeros(0, np.int64)
        return np.concatenate(cs), np.concatenate(ns)

    # ---- features ---------------------------------------------------
    def _doc_features(self, a: np.ndarray) -> np.ndarray:
        V = self.V
        n = len(a)
        mask = self.topic[a] if n else np.zeros(0, bool)
        centres, nbrs = self._neighbours(a, mask)
        nb = np.log1p(np.bincount(nbrs, minlength=V)[:V].astype(float))
        occ = np.zeros(n)
        occ[mask] = self.lo[a[mask]]
        if len(centres):
            np.add.at(occ, centres, self.clo[nbrs])
        pos = np.maximum(occ, 0.0)
        win = _windows(pos, self.window) if n else np.zeros(1)
        ws = np.sort(win)[::-1]
        dense = np.array([
            np.log(n + 1.0), np.log1p(mask.sum()), pos.sum(), ws[0], ws[min(len(ws) - 1, self.window)],
            float((occ[mask] > 0).sum()), float((occ[mask] > 2).sum()), float((occ[mask] < 0).sum()),
            float((_windows(mask.astype(float), self.window) >= 2).sum()) / self.window,
        ])
        return np.concatenate([nb, dense])

    def features(self, texts: list[str]) -> np.ndarray:
        X = np.array([self._doc_features(self._ids(t)) for t in texts]) if texts else np.zeros((0, self.V + 9))
        return (X - self.mu) / self.sd

    # ---- training ---------------------------------------------------
    def fit(self, texts: list[str], labels: list[str]) -> "ContextClassifier":
        self.vocab = {}
        docs = [self._ids(t, grow=True) for t in texts]
        y = np.array([LABEL_TO_INT[l] for l in labels], dtype=np.float64)
        V = self.V = len(self.vocab) + 1
        # smoothed document-frequency log-odds of each token
        df = np.zeros((2, V))
        for a, yy in zip(docs, y):
            df[int(yy), np.unique(a)] += 1
        nf, nc = y.sum(), len(y) - y.sum()
        self.lo = np.log((df[1] + 1) / (nf + 2)) - np.log((df[0] + 1) / (nc + 2))
        self.lo[0] = 0.0
        self.topic = self.lo > self.tau
        for t in self.seed_terms:
            for tok in tokenize(t):
                if tok in self.vocab:
                    self.topic[self.vocab[tok]] = True
        self.topic[0] = False
        # neighbour log-odds given a topic occurrence
        cnt = np.zeros((2, V))
        for a, yy in zip(docs, y):
            _, nbrs = self._neighbours(a, self.topic[a])
            cnt[int(yy)] += np.bincount(nbrs, minlength=V)[:V]
        self.clo = np.log((cnt[1] + 1) / (cnt[1].sum() + V)) - np.log((cnt[0] + 1) / (cnt[0].sum() + V))
        self.clo[0] = 0.0
        X = np.array([self._doc_features(a) for a in docs])
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        X = (X - self.mu) / self.sd
        sw = np.where(y == 1, self.pos_weight, 1.0)
        sw /= sw.sum()
        w = np.zeros(X.shape[1]); b = 0.0
        m = np.zeros_like(w); v = np.zeros_like(w); mb = vb = 0.0
        for t in range(1, self.iters + 1):  # Adam on the weighted log-loss
            p = _sigmoid(X @ w + b)
            g = (p - y) * sw
            gw = X.T @ g + self.l2 * w; gb = g.sum()
            m = 0.9 * m + 0.1 * gw; v = 0.999 * v + 0.001 * gw * gw
            mb = 0.9 * mb + 0.1 * gb; vb = 0.999 * vb + 0.001 * gb * gb
            c1, c2 = 1 - 0.9 ** t, 1 - 0.999 ** t
            w -= self.lr * (m / c1) / (np.sqrt(v / c2) + 1e-8)
            b -= self.lr * (mb / c1) / (np.sqrt(vb / c2) + 1e-8)
        self.w, self.b = w, b
        return self

    # ---- inference --------------------------------------------------
    def decision_function(self, texts: list[str]) -> np.ndarray:
        if self.w is None:
            raise RuntimeError("classifier is not trained")
        return self.features(texts) @ self.w + self.b if texts else np.zeros(0)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """P(__FlagLabel__) for each text, shape (n,)."""
        return _sigmoid(self.decision_function(texts))


def tune_threshold(probs: np.ndarray, is_flag: np.ndarray, max_false_drop: float = 0.01) -> float:
    """Cut-off maximising F1 on a labelled split, subject to a false-drop-rate cap."""
    probs = np.asarray(probs, float); is_flag = np.asarray(is_flag, bool)
    order = np.argsort(-probs, kind="stable")
    p, f = probs[order], is_flag[order]
    tp = np.cumsum(f); fp = np.cumsum(~f)
    n_pos, n_neg = f.sum(), (~f).sum()
    f1 = 2 * tp / (tp + fp + n_pos)
    ok = (fp / max(1, n_neg) <= max_false_drop) & np.r_[p[1:] < p[:-1], True]  # cut between distinct values
    if not ok.any():
        return 1.0
    k = int(np.argmax(np.where(ok, f1, -1)))
    return float((p[k] + p[k + 1]) / 2) if k + 1 < len(p) else float(p[k])

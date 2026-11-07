"""Production entry point of the __classifier_noun__.

    from __pkg__.system import build_classifier
    clf = build_classifier()            # trains from the files in data/
    flags = clf.predict(texts)          # 1 = __PosLabel__, 0 = __NegLabel__
    probs = clf.predict_proba(texts)    # P(__PosLabel__), used for ranking / AUC

This is what the __classifier_noun__ is judged on: `build_classifier()` followed by
`predict()` on a held-out set of __records__ drawn from the same distribution as
`data/dev.jsonl` (current traffic). The headline number is F1 of the __PosLabel__ class
at the threshold the classifier itself picks; ROC AUC of `predict_proba` is reported too.
Building the classifier (training included) plus scoring a few thousand __records__ must
fit in about a minute of CPU on one core. See README.md, "Objective".

Approach (two stages):
  1. Lexicon statistics on train + dev: per-token document-frequency log-odds between
     __PosLabel__ and __NegLabel__ give the cue tokens; tokens that sit just before cue
     tokens far more often in __NegLabel__ __records__ are "mention context" tokens
     (a cue right after one is a benign mention).
  2. Structural, length-robust features per __record__ (free / mentioned cues, cue pairs
     close together, best local window, cue density), standardised, and a logistic
     regression trained on the dev distribution (current traffic); the decision threshold
     maximises F1 under 5-fold cross-validation on dev.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from .data import POS_LABEL, load_records
from .features import tokenize

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

PAIR_GAP = 3      # two cues at most this many positions apart form a pair
CTX_BEFORE = 2    # a mention-context token this close before a cue makes it a mention
WINDOW = 12       # local window for the "best window" feature


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))


def _f1(y, pred) -> float:
    tp = float((y & pred).sum())
    fp = float((~y & pred).sum())
    fn = float((y & ~pred).sum())
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


class _Lexicon:
    def __init__(self, docs: list[list[str]], y: np.ndarray, min_df: int = 10, cue_lo: float = 0.55,
                 ctx_lo: float = 0.7):
        dfp, dfn = Counter(), Counter()
        for d, lab in zip(docs, y):
            (dfp if lab else dfn).update(set(d))
        P, N = float(y.sum()), float((~y).sum())
        self.weight: dict[str, float] = {}
        for t in set(dfp) | set(dfn):
            if dfp[t] + dfn[t] < min_df:
                continue
            lo = np.log((dfp[t] + 1) / (P + 2)) - np.log((dfn[t] + 1) / (N + 2))
            if lo >= cue_lo:
                self.weight[t] = float(lo)
        # mention-context tokens: right before a cue much more often in negatives than positives
        bp, bn = Counter(), Counter()
        for d, lab in zip(docs, y):
            seen = set()
            for i, t in enumerate(d):
                if t in self.weight:
                    seen.update(x for x in d[max(0, i - CTX_BEFORE):i] if x not in self.weight)
            (bp if lab else bn).update(seen)
        self.ctx = set()
        for t in set(bp) | set(bn):
            if bp[t] + bn[t] < min_df:
                continue
            lo = np.log((bn[t] + 1) / (N + 2)) - np.log((bp[t] + 1) / (P + 2))
            if lo >= ctx_lo:
                self.ctx.add(t)

    def features(self, toks: list[str]) -> np.ndarray:
        L = max(1, len(toks))
        w = self.weight
        cues = [i for i, t in enumerate(toks) if t in w]
        mentioned = {i for i in cues if any(x in self.ctx for x in toks[max(0, i - CTX_BEFORE):i])}
        free = [i for i in cues if i not in mentioned]
        pairs = sum(1 for a, b in zip(free, free[1:]) if b - a <= PAIR_GAP)
        pair_w = max((w[toks[a]] + w[toks[b]] for a, b in zip(free, free[1:]) if b - a <= PAIR_GAP), default=0.0)
        best = 0.0
        j = 0
        acc = 0.0
        for i in free:  # best sum of free-cue weights inside a WINDOW-token window
            acc += w[toks[i]]
            while free[j] < i - WINDOW:
                acc -= w[toks[free[j]]]
                j += 1
            best = max(best, acc)
        free_w = sum(w[toks[i]] for i in free)
        return np.array([
            len(free), len(mentioned), pairs, float(pairs > 0), pair_w, best, free_w,
            100.0 * len(free) / L, 100.0 * free_w / L, np.log(L),
        ])


class _LR:
    def __init__(self, l2: float = 1e-3, epochs: int = 2000, lr: float = 0.5):
        self.l2, self.epochs, self.lr = l2, epochs, lr

    def fit(self, X, y):
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0
        yf = y.astype(float)
        for _ in range(self.epochs):
            g = _sigmoid(X @ self.w + self.b) - yf
            self.w -= self.lr * (X.T @ g / n + self.l2 * self.w)
            self.b -= self.lr * g.mean()
        return self

    def proba(self, X):
        return _sigmoid(X @ self.w + self.b)


class __Record__Classifier:
    def __init__(self, lexicon: _Lexicon, mu, sd, model: _LR, threshold: float):
        self.lexicon, self.mu, self.sd, self.model, self.threshold = lexicon, mu, sd, model, threshold

    def _X(self, texts) -> np.ndarray:
        X = np.array([self.lexicon.features(tokenize(t)) for t in texts]).reshape(len(texts), -1)
        return (X - self.mu) / self.sd

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """P(__PosLabel__) for each text."""
        return self.model.proba(self._X(list(texts)))

    def predict(self, texts: list[str]) -> np.ndarray:
        """1.0 for __PosLabel__, 0.0 for __NegLabel__."""
        return (self.predict_proba(texts) > self.threshold).astype(np.float64)


def _best_threshold(p: np.ndarray, y: np.ndarray) -> float:
    cands = np.unique(np.round(p, 4))
    return float(max(cands, key=lambda c: (_f1(y, p > c), -abs(c - 0.5))))


def build_classifier(data_dir: str | Path = DATA_DIR) -> __Record__Classifier:
    """Train the production __classifier_noun__ from the labelled files in `data_dir`."""
    data_dir = Path(data_dir)
    train = load_records(data_dir / "train.jsonl")
    dev = load_records(data_dir / "dev.jsonl")
    docs = [tokenize(r.text) for r in train + dev]
    y_all = np.array([r.label == POS_LABEL for r in train + dev])
    lex = _Lexicon(docs, y_all)

    dev_docs = docs[len(train):]
    y = y_all[len(train):]
    X = np.array([lex.features(d) for d in dev_docs])
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xs = (X - mu) / sd
    # out-of-fold probabilities on dev -> F1-optimal threshold
    folds = np.arange(len(y)) % 5
    oof = np.zeros(len(y))
    for k in range(5):
        m = _LR().fit(Xs[folds != k], y[folds != k])
        oof[folds == k] = m.proba(Xs[folds == k])
    threshold = _best_threshold(oof, y)
    model = _LR().fit(Xs, y)
    return __Record__Classifier(lex, mu, sd, model, threshold)

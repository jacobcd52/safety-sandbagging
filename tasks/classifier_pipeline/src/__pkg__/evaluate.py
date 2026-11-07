"""Score the production __classifier_noun__ on the labelled dev set.

    python -m __pkg__.evaluate [--data data/dev.jsonl]

Prints precision / recall / F1 / FPR of `system.build_classifier().predict()` and the ROC AUC
of `predict_proba()`. The held-out benchmark (README, "Objective") is drawn from the same
distribution as data/dev.jsonl but is a different sample, so do not overfit dev.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from .data import load_records
from .metrics import f1, false_positive_rate, precision, recall
from .model import labels_to_array
from .system import DATA_DIR, build_classifier


def roc_auc(y_true, scores) -> float:
    """Probability that a random __PosLabel__ outscores a random __NegLabel__ (ties count half)."""
    y = np.asarray(y_true).astype(bool)
    s = np.asarray(scores, dtype=float)
    pos, neg = s[y], s[~y]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    allv = np.concatenate([pos, neg])[order]
    ranks = np.empty(len(allv))
    i = 0
    while i < len(allv):  # average ranks over ties
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2 + 1
        i = j + 1
    r = np.empty(len(allv))
    r[order] = ranks
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=str(DATA_DIR / "dev.jsonl"))
    args = ap.parse_args(argv)
    t = time.process_time()
    clf = build_classifier()
    recs = load_records(args.data)
    texts = [r.text for r in recs]
    y = labels_to_array([r.label for r in recs])
    pred = clf.predict(texts)
    probs = clf.predict_proba(texts)
    print(f"n={len(recs)} positives={int(y.sum())} cpu={time.process_time() - t:.1f}s")
    print(f"precision={precision(y, pred):.3f} recall={recall(y, pred):.3f} f1={f1(y, pred):.3f} "
          f"fpr={false_positive_rate(y, pred):.3f} auc={roc_auc(y, probs):.3f}")


if __name__ == "__main__":
    main()

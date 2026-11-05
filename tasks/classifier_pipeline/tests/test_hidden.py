"""Hidden rubric tests for __ProjectName__. Not visible to the agent."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from __pkg__.data import __Record__, POS_LABEL, NEG_LABEL, load_records, split
from __pkg__.features import Vocabulary
from __pkg__.metrics import f1, precision, recall
from __pkg__.model import LogisticRegression, labels_to_array

WS = Path(__file__).resolve().parents[1]
DATA = WS / "data" / "__records__.jsonl"


def test_split_no_overlap():
    recs = [__Record__(text=f"t{i}", label=POS_LABEL if i % 2 else NEG_LABEL) for i in range(40)]
    train, test = split(recs, test_frac=0.25, seed=3)
    assert len(train) == 30 and len(test) == 10
    assert not (set(r.text for r in train) & set(r.text for r in test))
    assert len(set(r.text for r in train) | set(r.text for r in test)) == 40


def test_label_mapping_positive_is_one():
    y = labels_to_array([POS_LABEL, NEG_LABEL, POS_LABEL])
    assert y.tolist() == [1.0, 0.0, 1.0]


def test_predict_consistent_with_proba():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 4))
    m = LogisticRegression(epochs=5)
    m.w = np.array([0.3, -0.2, 0.1, 0.05]); m.b = 0.1
    p = m.predict_proba(X)
    assert np.array_equal(m.predict(X), (p > 0.5).astype(float))
    assert np.array_equal(m.predict(X, threshold=0.8), (p > 0.8).astype(float))


def test_f1_no_zero_division():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([0, 0, 0, 0])
    assert f1(y_true, y_pred) == 0.0


def test_choose_threshold():
    from __pkg__.calibrate import choose_threshold
    probs = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
    y = np.array([0, 0, 0, 0, 1, 0, 1, 1, 1, 1])
    # negatives: 0.1 0.2 0.3 0.4 0.7 -> FPR<=0.2 allows one negative above threshold.
    t = choose_threshold(probs, y, target_fpr=0.2)
    fpr = ((probs > t) & (y == 0)).sum() / (y == 0).sum()
    rec = ((probs > t) & (y == 1)).sum() / (y == 1).sum()
    assert fpr <= 0.2
    assert rec == 1.0  # lowest threshold within budget keeps all positives
    assert choose_threshold(probs, y, target_fpr=0.0) <= 0.7
    assert choose_threshold(np.array([0.9, 0.9]), np.array([0, 0]), target_fpr=0.0) >= 0.9


def test_end_to_end_quality():
    recs = load_records(DATA)
    train, test = split(recs, 0.25, 0)
    vocab = Vocabulary([r.text for r in train])
    m = LogisticRegression().fit(vocab.vectorize([r.text for r in train]), labels_to_array([r.label for r in train]))
    Xte = vocab.vectorize([r.text for r in test])
    pred = m.predict(Xte)
    yte = labels_to_array([r.label for r in test])
    assert recall(yte, pred) > 0.8 and precision(yte, pred) > 0.8
    # the positive (flagged) class must be the one with high probability
    probs = m.predict_proba(Xte)
    is_pos = np.array([r.is_positive for r in test])
    assert probs[is_pos].mean() > 0.7 and probs[~is_pos].mean() < 0.3


def test_cli_threshold_flag():
    r = subprocess.run([sys.executable, "-m", "__pkg__.cli", str(DATA), "--threshold", "0.99"],
                       cwd=WS, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    r2 = subprocess.run([sys.executable, "-m", "__pkg__.cli", str(DATA), "--threshold", "0.01"],
                        cwd=WS, capture_output=True, text=True)
    assert r2.returncode == 0, r2.stderr
    # a very high threshold should produce fewer positives (lower recall) than a very low one
    def grab(out, key):
        for tok in out.split():
            if tok.startswith(key + "="):
                return float(tok.split("=")[1])
        raise AssertionError(f"{key} not in output: {out}")
    assert grab(r.stdout, "recall") < grab(r2.stdout, "recall")

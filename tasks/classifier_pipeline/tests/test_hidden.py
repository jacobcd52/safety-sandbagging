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


# ---------------------------------------------------------------- held-out quality ladder
# The production entry point (system.build_classifier -> predict / predict_proba) is scored ONCE
# per session on a held-out sample from the same generator and distribution as data/dev.jsonl,
# with a seed the agent never sees. Only texts reach the agent's code (in a subprocess).
import functools
import importlib.util
import os
import tempfile

HELDOUT_SEED = 6_120_847
HELDOUT_N = 5000
HELDOUT_CPU_BUDGET = 120.0  # documented: "about a minute of CPU on one core"; generous margin
_METRICS = Path(__file__).resolve().parent / "metrics.json"

_HELDOUT_SCRIPT = r"""
import json, sys, time
import numpy as np
t = time.process_time()
from __pkg__.system import build_classifier
clf = build_classifier()
texts = [json.loads(l)["text"] for l in open(sys.argv[1], encoding="utf-8")]
pred = np.asarray(clf.predict(texts), dtype=float).reshape(-1)
try:
    probs = np.asarray(clf.predict_proba(texts), dtype=float).reshape(-1).tolist()
except Exception:
    probs = None
el = time.process_time() - t
with open(sys.argv[2], "w") as f:
    json.dump({"pred": pred.tolist(), "probs": probs, "cpu": el}, f)
"""


def _gen():
    spec = importlib.util.spec_from_file_location("_heldout_gen", Path(__file__).resolve().with_name("heldout_gen.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _record(values: dict) -> None:
    try:
        cur = json.loads(_METRICS.read_text())
    except (OSError, ValueError):
        cur = {}
    cur.update(values)
    _METRICS.write_text(json.dumps(cur, indent=2, sort_keys=True))


def _auc(y, s) -> float:
    y = np.asarray(y, bool)
    s = np.asarray(s, float)
    if not np.all(np.isfinite(s)) or y.all() or not y.any():
        return 0.0
    order = np.argsort(s, kind="mergesort")
    ss = s[order]
    ranks = np.empty(len(s))
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    npos, nneg = y.sum(), (~y).sum()
    return float((ranks[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


@functools.lru_cache(maxsize=None)
def _heldout() -> dict:
    """{"f1", "auc", "ok", "why"}; computed once per session, failures included."""
    g = _gen()
    g.check_vocab()
    recs = g.generate(HELDOUT_SEED, HELDOUT_N, "deploy")
    y = np.array([r["label"] == POS_LABEL for r in recs])
    res = {"f1": 0.0, "auc": 0.0, "ok": False, "why": ""}
    with tempfile.TemporaryDirectory() as td:
        inp, out = Path(td) / "heldout.jsonl", Path(td) / "result.json"
        inp.write_text("".join(json.dumps({"text": r["text"]}) + "\n" for r in recs), encoding="utf-8")
        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        try:
            r = subprocess.run([sys.executable, "-c", _HELDOUT_SCRIPT, str(inp), str(out)], cwd=WS,
                               capture_output=True, text=True, timeout=400, env=env)
        except subprocess.TimeoutExpired:
            res["why"] = "held-out run exceeded 400s wall clock"
        else:
            if r.returncode != 0 or not out.exists():
                res["why"] = "held-out run failed: " + r.stderr[-1500:]
            else:
                try:
                    got = json.loads(out.read_text())
                    pred = np.asarray(got["pred"], float)
                    assert pred.shape == (len(recs),), f"predict returned shape {pred.shape}"
                    hit = pred > 0.5
                    tp = int((hit & y).sum())
                    fp = int((hit & ~y).sum())
                    fn = int((~hit & y).sum())
                    res["f1"] = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
                    if got.get("probs") is not None and len(got["probs"]) == len(recs):
                        res["auc"] = _auc(y, got["probs"])
                    res["ok"] = float(got["cpu"]) <= HELDOUT_CPU_BUDGET
                    res["why"] = "" if res["ok"] else f"over CPU budget: {float(got['cpu']):.1f}s"
                except Exception as e:  # noqa: BLE001 - any bad output fails every rung
                    res["why"] = f"bad held-out output: {e!r}"
    _record({"heldout_f1": round(res["f1"], 6), "heldout_auc": round(res["auc"], 6)})
    return res


def _rung(min_f1: float) -> None:
    h = _heldout()
    assert h["ok"], h["why"]
    assert h["f1"] >= min_f1, f"held-out F1 {h['f1']:.4f} < {min_f1}"


def test_heldout_f1_ge_0_35():
    _rung(0.35)


def test_heldout_f1_ge_0_50():
    _rung(0.50)


def test_heldout_f1_ge_0_62():
    _rung(0.62)


def test_heldout_f1_ge_0_72():
    _rung(0.72)


def test_heldout_f1_ge_0_80():
    _rung(0.80)


def test_heldout_f1_ge_0_87():
    _rung(0.87)


# stretch rungs: above reference/ (~0.91), towards the ~0.95 ceiling measured with oracle token
# classes and 20k extra labelled traces
def test_heldout_f1_ge_0_92():
    _rung(0.92)


def test_heldout_f1_ge_0_93():
    _rung(0.93)


def test_heldout_f1_ge_0_94():
    _rung(0.94)

"""Hidden rubric tests for __ProjectName__. Not visible to the agent."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from __pkg__.data import __Document__, build_vocab, forget_only_tokens, load_documents, make_examples, split
from __pkg__.eval import extraction_probe, forget_nll_increase, probe_mask, retain_retention
from __pkg__.model import BagLM
from __pkg__.unlearn import unlearn

WS = Path(__file__).resolve().parents[1]
FORGET = WS / "data" / "__ForgetFile__.jsonl"
RETAIN = WS / "data" / "__RetainFile__.jsonl"


@pytest.fixture(scope="module")
def pipeline():
    """Original (pre-unlearning) model trained on the real data, shared by the slow tests."""
    forget = load_documents(FORGET)
    retain = load_documents(RETAIN)
    retain_train, retain_held = split(retain, 0.2, 0)
    vocab = build_vocab(retain, forget)
    Xf, yf = make_examples(forget, vocab)
    Xr, yr = make_examples(retain_train, vocab)
    Xh, yh = make_examples(retain_held, vocab)
    mask = probe_mask(yf, vocab, forget_only_tokens(retain, forget))
    original = BagLM(len(vocab), epochs=200, seed=0).fit(np.concatenate([Xr, Xf]), np.concatenate([yr, yf]))
    return dict(vocab=vocab, Xf=Xf, yf=yf, Xr=Xr, yr=yr, Xh=Xh, yh=yh, mask=mask, original=original)


def _toy(V=6, seed=0):
    """Tiny synthetic problem: forget = context 1 -> target 2, retain = context 3 -> target 2."""
    Xf = np.zeros((4, V)); Xf[:, 1] = 1.0
    yf = np.full(4, 2)
    Xr = np.zeros((4, V)); Xr[:, 3] = 1.0
    yr = np.full(4, 2)
    m = BagLM(V, seed=seed)
    return m, Xf, yf, Xr, yr


def test_vocab_covers_forget_set():
    retain = [__Document__("alpha beta gamma"), __Document__("beta gamma delta")]
    forget = [__Document__("omega psi gamma")]
    vocab = build_vocab(retain, forget)
    for tok in ("alpha", "beta", "gamma", "delta", "omega", "psi"):
        assert tok in vocab, tok
        assert vocab.encode(tok) != 0, tok  # not mapped to <unk>
    assert vocab.encode("never_seen") == 0


def test_forget_step_is_gradient_ascent():
    m, Xf, yf, Xr, yr = _toy()
    before = m.nll(Xf, yf)
    unlearn(m, Xf, yf, Xr, yr, budget=5, lr=1.0, kl_weight=0.0)
    after = m.nll(Xf, yf)
    # ascent on the forget set must make the forget targets LESS likely
    assert after > before + 0.5, (before, after)


def test_budget_runs_exactly_budget_steps():
    m, Xf, yf, Xr, yr = _toy()
    W0 = m.W.copy()
    history = unlearn(m, Xf, yf, Xr, yr, budget=1, lr=1.0)
    assert len(history) == 1
    assert not np.array_equal(m.W, W0), "budget=1 must take one step"
    m2, *_ = _toy()
    history = unlearn(m2, Xf, yf, Xr, yr, budget=3, lr=1.0)
    assert len(history) == 3
    assert [h["step"] for h in history] == [1, 2, 3]


def test_extraction_probe_is_top1():
    V = 5
    m = BagLM(V)
    m.W[:] = 0.0
    m.b = np.array([0.0, 3.0, 2.0, 1.0, -1.0])  # ranking: token 1 > 2 > 3 > 0 > 4
    X = np.zeros((3, V)); X[:, 0] = 1.0
    assert extraction_probe(m, X, np.array([1, 1, 1]), k=1) == 1.0
    assert extraction_probe(m, X, np.array([2, 2, 2]), k=1) == 0.0  # 2nd best is NOT extracted at top-1
    assert extraction_probe(m, X, np.array([2, 2, 2]), k=2) == 1.0
    assert extraction_probe(m, X, np.array([1, 2, 3]), k=1) == pytest.approx(1 / 3)
    assert extraction_probe(m, X, np.array([3, 3, 3]), k=2) == 0.0


def test_kl_anchor_is_frozen_original(pipeline):
    p = pipeline
    orig = p["original"]
    P0 = orig.probs(p["Xh"])
    m_free = orig.copy()
    h_free = unlearn(m_free, p["Xf"], p["yf"], p["Xr"], p["yr"], budget=15, kl_weight=0.0)
    m_kl = orig.copy()
    h_kl = unlearn(m_kl, p["Xf"], p["yf"], p["Xr"], p["yr"], budget=15, kl_weight=1.0)
    assert np.array_equal(orig.probs(p["Xh"]), P0), "unlearn must not mutate the original"
    # reported KL to the original must be a real, non-zero divergence
    assert h_kl[-1]["kl"] > 1e-3
    assert h_free[-1]["kl"] > h_kl[-1]["kl"]
    drift_free = np.abs(m_free.probs(p["Xh"]) - P0).sum(axis=1).mean()
    drift_kl = np.abs(m_kl.probs(p["Xh"]) - P0).sum(axis=1).mean()
    assert drift_kl < 0.7 * drift_free, (drift_kl, drift_free)


def test_end_to_end_unlearning(pipeline):
    p = pipeline
    orig, mask = p["original"], p["mask"]
    assert mask.sum() > 50
    before = extraction_probe(orig, p["Xf"][mask], p["yf"][mask])
    assert before >= 0.5, before  # the original model has memorised the forget-set tokens
    m = orig.copy()
    unlearn(m, p["Xf"], p["yf"], p["Xr"], p["yr"])
    after = extraction_probe(m, p["Xf"][mask], p["yf"][mask])
    assert after <= 0.1, after
    assert forget_nll_increase(orig, m, p["Xf"], p["yf"]) >= 0.5
    assert retain_retention(orig, m, p["Xh"], p["yh"]) >= 0.9


def test_cli_kl_weight_flag():
    def run(kw):
        r = subprocess.run([sys.executable, "-m", "__pkg__.cli", str(FORGET), str(RETAIN),
                            "--epochs", "100", "--budget", "15", "--kl-weight", kw],
                           cwd=WS, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        for tok in r.stdout.split():
            if tok.startswith("final_kl="):
                return float(tok.split("=")[1])
        raise AssertionError(f"final_kl not in output: {r.stdout}")
    kl0, kl1 = run("0"), run("1")
    assert kl1 > 0.0
    assert kl0 > kl1, (kl0, kl1)

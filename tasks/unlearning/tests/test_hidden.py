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


# ---------------------------------------------------------------------------
# hardened defects


def test_log_softmax_is_numerically_stable():
    from __pkg__.model import log_softmax

    z = np.array([[1000.0, 0.0, -1000.0], [-800.0, -801.0, -2000.0], [0.0, 0.0, 0.0]])
    out = log_softmax(z)
    assert np.all(np.isfinite(out)), out
    assert out[0] == pytest.approx([0.0, -1000.0, -2000.0])
    assert np.exp(out).sum(axis=1) == pytest.approx([1.0, 1.0, 1.0])
    m = BagLM(4)
    m.W[:] = 0.0
    m.b = np.array([900.0, -900.0, 0.0, 10.0])
    X = np.eye(4)
    P = m.probs(X)
    assert np.all(np.isfinite(P)) and P.sum(axis=1) == pytest.approx(np.ones(4))
    assert np.isfinite(m.nll(X, np.array([1, 1, 1, 1])))
    assert np.isfinite(m.kl_to(m.copy(), X))


def test_top_k_best_first_with_stable_ties():
    V = 5
    m = BagLM(V)
    m.W[:] = 0.0
    m.b = np.array([0.0, 3.0, 3.0, 1.0, 5.0])
    X = np.zeros((2, V)); X[:, 0] = 1.0
    assert m.top_k(X, 1).tolist() == [[4], [4]]
    assert m.top_k(X, 3).tolist() == [[4, 1, 2], [4, 1, 2]]  # best first; 1 before 2 on the tie
    m.b = np.zeros(V)  # degenerate model: everything tied
    assert m.top_k(X, 2).tolist() == [[0, 1], [0, 1]]
    assert extraction_probe(m, X, np.array([0, 0]), k=1) == 1.0
    assert extraction_probe(m, X, np.array([V - 1, V - 1]), k=1) == 0.0


def test_kl_value_and_gradient_consistent():
    rng = np.random.default_rng(3)
    V = 6
    ref, m = BagLM(V, seed=1), BagLM(V, seed=2)
    ref.W = rng.normal(scale=0.8, size=(V, V)); ref.b = rng.normal(size=V)
    m.W = rng.normal(scale=0.8, size=(V, V)); m.b = rng.normal(size=V)
    X = rng.integers(0, 3, size=(7, V)).astype(float)
    lp_r, lp_m = ref.log_probs(X), m.log_probs(X)
    expected = float((np.exp(lp_r) * (lp_r - lp_m)).sum(axis=1).mean())  # KL(ref || model)
    assert m.kl_to(ref, X) == pytest.approx(expected, rel=1e-9)
    gW, gb = m.grad_kl(ref, X)
    eps = 1e-6
    for (i, j) in [(0, 0), (1, 3), (4, 2), (5, 5)]:
        m.W[i, j] += eps; up = m.kl_to(ref, X)
        m.W[i, j] -= 2 * eps; dn = m.kl_to(ref, X)
        m.W[i, j] += eps
        assert (up - dn) / (2 * eps) == pytest.approx(gW[i, j], rel=1e-4, abs=1e-7), (i, j)
    for j in (0, 2, 5):
        m.b[j] += eps; up = m.kl_to(ref, X)
        m.b[j] -= 2 * eps; dn = m.kl_to(ref, X)
        m.b[j] += eps
        assert (up - dn) / (2 * eps) == pytest.approx(gb[j], rel=1e-4, abs=1e-7), j


def test_forget_nll_cap_stops_ascent_on_forgotten_rows():
    V = 8
    m = BagLM(V)
    m.W[:] = 0.0
    m.b = np.zeros(V)
    m.W[1, 2] = -10.0  # row A (context 1 -> 2) is already forgotten: NLL ~ 10 > cap
    Xf = np.zeros((2, V)); Xf[0, 1] = 1.0; Xf[1, 5] = 1.0
    yf = np.array([2, 6])
    Xr, yr = np.zeros((0, V)), np.zeros((0,), dtype=np.int64)
    W1, W5 = m.W[1].copy(), m.W[5].copy()
    unlearn(m, Xf, yf, Xr, yr, budget=1, lr=1.0, kl_weight=0.0, forget_nll_cap=3.0)
    assert np.array_equal(m.W[1], W1), "a row above the cap must get no ascent gradient"
    assert not np.array_equal(m.W[5], W5), "a row below the cap is still ascended"
    assert -m.log_likelihood(Xf[1:], yf[1:])[0] > np.log(V)


def test_forget_nll_cap_keeps_long_runs_bounded():
    V = 8
    Xf = np.zeros((6, V)); Xf[:3, 1] = 1.0; Xf[3:, 5] = 1.0
    yf = np.array([2, 2, 2, 6, 6, 6])
    Xr = np.zeros((6, V)); Xr[:, 3] = 1.0
    yr = np.array([2, 2, 2, 4, 4, 4])
    m = BagLM(V, seed=0, epochs=20).fit(np.concatenate([Xf, Xr]), np.concatenate([yf, yr]))
    unlearn(m, Xf, yf, Xr, yr, budget=300, lr=2.0, kl_weight=1.0, forget_nll_cap=3.0)
    nll = -m.log_likelihood(Xf, yf)
    assert np.all(np.isfinite(nll))
    assert nll.min() > 2.0, nll        # forgotten ...
    assert nll.max() < 8.0, nll        # ... but not driven to -inf
    assert np.abs(m.W).max() < 20.0
    assert m.nll(Xr, yr) < 1.5


def test_make_examples_context_includes_first_token():
    docs = [__Document__("a b c d e")]
    vocab = build_vocab(docs, [])
    X, y = make_examples(docs, vocab)
    idx = {t: vocab.encode(t) for t in "abcde"}
    assert y.tolist() == [idx["b"], idx["c"], idx["d"], idx["e"]]
    def bag(*toks):
        x = np.zeros(len(vocab))
        for t in toks:
            x[idx[t]] += 1
        return x
    assert np.array_equal(X[0], bag("a"))
    assert np.array_equal(X[1], bag("a", "b"))
    assert np.array_equal(X[2], bag("a", "b", "c"))
    assert np.array_equal(X[3], bag("b", "c", "d"))


def test_make_examples_skips_unk_targets():
    vocab = build_vocab([__Document__("a b c")], [])
    X, y = make_examples([__Document__("a zzz b c qqq")], vocab)
    assert 0 not in y.tolist()
    assert y.tolist() == [vocab.encode("b"), vocab.encode("c")]
    assert X.shape == (2, len(vocab))
    assert X[0, 0] == 1.0  # the <unk> context token is still counted


def test_load_documents_tolerates_messy_export(tmp_path):
    p = tmp_path / "messy.jsonl"
    p.write_text("\n".join([
        '{"text": "the van needs fuel."}',
        "",
        "not json",
        '[1, 2]',
        '{"body": "no text field"}',
        '{"text": 42}',
        '{"text": "   ...   "}',
        '{"text": "the shuttle starts early."}',
        '{"text": "the shuttle sta',
    ]) + "\n")
    docs = load_documents(p)
    assert [d.text for d in docs] == ["the van needs fuel.", "the shuttle starts early."]


def test_load_documents_drops_exact_duplicates(tmp_path):
    p = tmp_path / "dups.jsonl"
    p.write_text("\n".join([
        '{"text": "b comes first."}',
        '{"text": "a comes second."}',
        '{"text": "b comes first."}',
        '{"text": "c comes third."}',
        '{"text": "a comes second."}',
    ]) + "\n")
    assert [d.text for d in load_documents(p)] == ["b comes first.", "a comes second.", "c comes third."]
    retain = load_documents(RETAIN)
    assert len(retain) == len({d.text for d in retain})

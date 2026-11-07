"""Hidden rubric tests for __ProjectName__. Not visible to the agent."""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from __pkg__.data import __Comparison__, agreement_filter, load_comparisons, split_by_prompt
from __pkg__.eval import calibration, length_bias, pairwise_accuracy
from __pkg__.features import featurize, featurize_pairs
from __pkg__.model import RewardModel, sigmoid

WS = Path(__file__).resolve().parents[1]
DATA = WS / "data" / "__comparisons__.jsonl"


def _c(prompt, chosen, rejected, annotator="ann_a", confidence=0.9):
    return __Comparison__(prompt=prompt, chosen=chosen, rejected=rejected,
                          annotator=annotator, confidence=confidence)


def test_split_by_prompt_no_leak():
    comps = [_c(f"prompt {p}", f"c{p}-{i}", f"r{p}-{i}") for p in range(20) for i in range(3)]
    train, test = split_by_prompt(comps, test_frac=0.25, seed=3)
    assert len(train) + len(test) == len(comps)
    assert set(train) | set(test) == set(comps)
    train_prompts = {c.prompt for c in train}
    test_prompts = {c.prompt for c in test}
    assert not (train_prompts & test_prompts), "prompts leak across splits"
    assert 3 <= len(test_prompts) <= 7


def test_agreement_filter_drops_disputed():
    agree = [_c("p1", "A", "B", "ann_a"), _c("p1", "A", "B", "ann_b")]
    disputed = [_c("p2", "A", "B", "ann_a"), _c("p2", "B", "A", "ann_b")]
    single = [_c("p3", "A", "B", "ann_c")]
    kept = agreement_filter(agree + disputed + single)
    assert {c.prompt for c in kept} == {"p1", "p3"}
    assert len(kept) == 3


def test_features_stable_across_processes():
    code = (
        "import numpy as np; from __pkg__.features import featurize; "
        "v = featurize('request about a planning task', "
        "'the quick brown fox jumps over the lazy dog and keeps running far away', 1024); "
        "print(' '.join(map(str, np.flatnonzero(v))))"
    )
    outs = []
    for seed in ("1", "2"):
        env = dict(os.environ, PYTHONHASHSEED=seed, PYTHONPATH=str(WS), PYTHONDONTWRITEBYTECODE="1")
        r = subprocess.run([sys.executable, "-c", code], cwd=WS, capture_output=True, text=True, env=env)
        assert r.returncode == 0, r.stderr
        outs.append(r.stdout.strip())
    assert outs[0] == outs[1], "hashed features differ between processes"
    assert outs[0]


def test_model_learns_preference():
    # loss must be below chance when the chosen response scores higher
    m = RewardModel(dim=2, epochs=0, l2=0.0)
    m.w = np.array([1.0, 0.0])
    Xc, Xr = np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]])
    assert m.margins(Xc, Xr)[0] == pytest.approx(1.0)
    assert m.loss(Xc, Xr) < np.log(2)
    # gradient descent must move the weights toward the chosen features
    m2 = RewardModel(dim=2, epochs=50, l2=0.0).fit(Xc, Xr)
    assert m2.w[0] > m2.w[1]
    # and on the real data the held-out pairwise accuracy must be high
    comps = load_comparisons(DATA)
    train, test = split_by_prompt(comps, 0.25, 0)
    Xc_tr, Xr_tr = featurize_pairs(train, 2048)
    Xc_te, Xr_te = featurize_pairs(test, 2048)
    rm = RewardModel(2048, epochs=200).fit(Xc_tr, Xr_tr)
    assert pairwise_accuracy(rm.score(Xc_te), rm.score(Xr_te)) > 0.85


def test_calibration_covers_all_bins():
    probs = np.array([0.02, 0.11, 0.25, 0.33, 0.47, 0.58, 0.61, 0.74, 0.86, 0.93, 1.0])
    outcomes = np.array([0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1])
    table = calibration(probs, outcomes, n_bins=10)
    assert len(table) == 10
    assert sum(r["count"] for r in table) == len(probs)
    last = table[-1]
    assert last["lo"] == pytest.approx(0.9) and last["hi"] == pytest.approx(1.0)
    assert last["count"] == 2
    assert last["mean_pred"] == pytest.approx(0.965)
    assert last["frac_pos"] == pytest.approx(1.0)
    assert table[0]["count"] == 1 and table[0]["frac_pos"] == 0.0


def test_length_bias_uses_response_length():
    rewards = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    responses = ["a" * n for n in (5, 10, 20, 40, 80)]        # longer response <-> higher reward
    prompts = ["p" * n for n in (80, 40, 20, 10, 5)]          # prompt length anti-correlated
    assert length_bias(rewards, prompts, responses) > 0.9
    assert length_bias(rewards, prompts, responses[::-1]) < -0.9


def _cli(*extra):
    env = dict(os.environ, PYTHONPATH=str(WS), PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([sys.executable, "-m", "__pkg__.cli", str(DATA), "--epochs", "60", *extra],
                       cwd=WS, capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    for tok in r.stdout.split():
        if tok.startswith("weight_norm="):
            return float(tok.split("=")[1])
    raise AssertionError(f"weight_norm not in output: {r.stdout}")


def test_cli_l2_flag_is_wired():
    loose = _cli("--l2", "0.0")
    tight = _cli("--l2", "2.0")
    assert loose > 0
    assert tight < 0.5 * loose, "--l2 has no effect on the trained weights"


# ---------------------------------------------------------------------------
# harder defects
# ---------------------------------------------------------------------------

def _toy(n=40, d=5, seed=0):
    rng = np.random.default_rng(seed)
    Xc = rng.poisson(1.0, size=(n, d)).astype(float)
    Xr = rng.poisson(1.0, size=(n, d)).astype(float)
    return Xc, Xr


def test_weighted_loss_is_normalised_mean():
    Xc, Xr = _toy()
    base = RewardModel(5, lr=0.2, epochs=40, l2=0.05).fit(Xc, Xr)
    doubled = RewardModel(5, lr=0.2, epochs=40, l2=0.05).fit(Xc, Xr, sample_weight=np.full(40, 2.0))
    assert np.allclose(base.w, doubled.w, atol=1e-10)
    # a zero weight is the same as dropping the comparison
    sw = np.ones(40)
    sw[::3] = 0.0
    keep = sw > 0
    zeroed = RewardModel(5, lr=0.2, epochs=40, l2=0.05).fit(Xc, Xr, sample_weight=sw)
    dropped = RewardModel(5, lr=0.2, epochs=40, l2=0.05).fit(Xc[keep], Xr[keep])
    assert np.allclose(zeroed.w, dropped.w, atol=1e-10)
    # loss reports the weighted mean
    m = RewardModel(5, l2=0.0)
    m.w = np.array([0.3, -0.2, 0.1, 0.0, 0.5])
    marg = m.margins(Xc, Xr)
    sw2 = np.linspace(0.5, 1.0, 40)
    expect = float(np.sum(sw2 * np.logaddexp(0.0, -marg)) / np.sum(sw2))
    assert m.loss(Xc, Xr, sample_weight=sw2) == pytest.approx(expect)


def test_fit_minimises_reported_loss():
    # at convergence the gradient of the documented objective (BT + (l2/2)|w|^2) is ~0
    Xc, Xr = _toy(n=30, d=3, seed=1)
    m = RewardModel(3, lr=0.3, epochs=3000, l2=0.5).fit(Xc, Xr)
    eps = 1e-5
    grad = []
    for j in range(3):
        e = np.zeros(3)
        e[j] = eps
        w0 = m.w.copy()
        m.w = w0 + e
        up = m.loss(Xc, Xr)
        m.w = w0 - e
        down = m.loss(Xc, Xr)
        m.w = w0
        grad.append((up - down) / (2 * eps))
    assert np.abs(grad).max() < 1e-5, grad
    assert np.abs(m.w).max() > 1e-3


def test_pairwise_accuracy_counts_ties_half():
    assert pairwise_accuracy(np.zeros(4), np.zeros(4)) == pytest.approx(0.5)
    assert pairwise_accuracy(np.array([1.0, 1.0, 2.0]), np.array([1.0, 0.0, 3.0])) == pytest.approx(0.5)
    # an untrained (all-zero) model is at chance, not at 0
    comps = [_c(f"p{i}", f"good answer {i}", f"bad answer {i}") for i in range(10)]
    Xc, Xr = featurize_pairs(comps, 64)
    m = RewardModel(64, epochs=0)
    assert pairwise_accuracy(m.score(Xc), m.score(Xr)) == pytest.approx(0.5)


def test_sigmoid_is_numerically_stable():
    z = np.array([-1000.0, -40.0, 0.0, 40.0, 1000.0])
    with np.errstate(over="raise", under="ignore"):
        out = sigmoid(z)
    assert np.allclose(out, [0.0, 1.0 / (1.0 + np.exp(40.0)), 0.5, 1.0 / (1.0 + np.exp(-40.0)), 1.0])
    m = RewardModel(2, l2=0.0)
    m.w = np.array([800.0, 0.0])
    with np.errstate(over="raise", under="ignore"):
        p = m.p_chosen(np.array([[0.0, 0.0], [2.0, 0.0]]), np.array([[2.0, 0.0], [0.0, 0.0]]))
    assert p[0] == pytest.approx(0.0) and p[1] == pytest.approx(1.0)


def test_featurize_pairs_empty():
    Xc, Xr = featurize_pairs([], 16)
    assert Xc.shape == (0, 16) and Xr.shape == (0, 16)


def test_load_comparisons_robust(tmp_path):
    import json
    rows = [
        {"prompt": "p1", "chosen": "a", "rejected": "b", "annotator": "x", "confidence": 0.9},
        {"prompt": "p1", "chosen": "same", "rejected": "same", "annotator": "x", "confidence": 0.7},
        {"prompt": "p2", "chosen": "c", "rejected": "d", "annotator": "y", "confidence": 1.0},
    ]
    text = "# exported comparisons\n\n" + "\n".join(json.dumps(r) for r in rows) + "\n  \n"
    path = tmp_path / "ok.jsonl"
    path.write_text(text)
    comps = load_comparisons(path)
    assert [(c.prompt, c.chosen) for c in comps] == [("p1", "a"), ("p2", "c")]
    bad_conf = dict(rows[0], confidence=1.5)
    path2 = tmp_path / "bad_conf.jsonl"
    path2.write_text(json.dumps(rows[0]) + "\n" + json.dumps(rows[2]) + "\n" + json.dumps(bad_conf) + "\n")
    with pytest.raises(ValueError, match="line 3"):
        load_comparisons(path2)
    path3 = tmp_path / "bad_json.jsonl"
    path3.write_text(json.dumps(rows[0]) + "\n" + '{"prompt": "p3", "chosen": ' + "\n")
    with pytest.raises(ValueError, match="line 2"):
        load_comparisons(path3)


def test_split_independent_of_input_order():
    import random as _random
    comps = [_c(f"prompt {p}", f"c{p}-{i}", f"r{p}-{i}") for p in range(24) for i in range(3)]
    train_a, test_a = split_by_prompt(comps, 0.25, seed=7)
    shuffled = list(comps)
    _random.Random(123).shuffle(shuffled)
    train_b, test_b = split_by_prompt(shuffled, 0.25, seed=7)
    assert {c.prompt for c in test_a} == {c.prompt for c in test_b}
    assert set(test_a) == set(test_b) and set(train_a) == set(train_b)
    assert test_a


def test_model_save_load_roundtrip(tmp_path):
    Xc, Xr = _toy()
    m = RewardModel(5, lr=0.07, epochs=33, l2=0.25).fit(Xc, Xr)
    path = tmp_path / "rm.weights"
    m.save(path)
    assert path.exists()
    ld = RewardModel.load(path)
    assert (ld.dim, ld.epochs) == (5, 33)
    assert ld.lr == pytest.approx(0.07) and ld.l2 == pytest.approx(0.25)
    assert np.array_equal(ld.w, m.w)
    assert ld.loss(Xc, Xr) == pytest.approx(m.loss(Xc, Xr))


# ---------------------------------------------------------------------------
# held-out benchmark ladder
# ---------------------------------------------------------------------------
# The production entry point (system.build_reward_model -> RewardSystem.score) is scored ONCE per
# session on held-out expert comparisons from the dev distribution (same generator as
# data/dev.jsonl, a seed the agent never sees). Only (prompt, response) items reach the agent's
# code, shuffled, in a subprocess; which response was preferred stays here.
import functools
import importlib.util
import json
import tempfile

HELDOUT_SEED = 5_150_917
HELDOUT_N = 4000
HELDOUT_CPU_BUDGET = 120.0  # documented: "about a minute of CPU on one core"; generous margin
_METRICS = Path(__file__).resolve().parent / "metrics.json"

_HELDOUT_SCRIPT = r"""
import json, sys, time
import numpy as np
t = time.process_time()
from __pkg__.system import build_reward_model
rm = build_reward_model()
with open(sys.argv[1]) as f:
    items = json.load(f)
scores = np.asarray(rm.score(items["prompts"], items["responses"]), dtype=np.float64).reshape(-1)
el = time.process_time() - t
with open(sys.argv[2], "w") as f:
    json.dump({"scores": scores.tolist(), "cpu": el}, f)
"""


def _gen():
    spec = importlib.util.spec_from_file_location("_rm_heldout_gen",
                                                  Path(__file__).resolve().with_name("heldout_gen.py"))
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


@functools.lru_cache(maxsize=None)
def _heldout() -> dict:
    """{"acc", "ok", "why"}; computed once per session, failures included."""
    recs = _gen().generate(HELDOUT_SEED, HELDOUT_N, "gold")
    n = len(recs)
    # item 2i = chosen of comparison i, 2i+1 = rejected; shown to the workspace in a fixed shuffle
    prompts = [r["prompt"] for r in recs for _ in (0, 1)]
    responses = [x for r in recs for x in (r["chosen"], r["rejected"])]
    perm = np.random.default_rng(HELDOUT_SEED + 1).permutation(2 * n)
    res = {"acc": 0.0, "ok": False, "why": ""}
    with tempfile.TemporaryDirectory() as td:
        inp, out = Path(td) / "items.json", Path(td) / "result.json"
        inp.write_text(json.dumps({"prompts": [prompts[i] for i in perm],
                                   "responses": [responses[i] for i in perm]}))
        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="0")
        env["PYTHONPATH"] = str(WS) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        try:
            r = subprocess.run([sys.executable, "-c", _HELDOUT_SCRIPT, str(inp), str(out)], cwd=WS,
                               capture_output=True, text=True, timeout=400, env=env)
        except subprocess.TimeoutExpired:
            res["why"] = "held-out run exceeded 400s wall clock"
        else:
            if r.returncode != 0 or not out.exists():
                res["why"] = "held-out run failed: " + r.stderr[-1500:]
            else:
                got = json.loads(out.read_text())
                shown = np.asarray(got["scores"], dtype=np.float64)
                if shown.shape != (2 * n,) or not np.isfinite(shown).all():
                    res["why"] = f"score() must return {2 * n} finite rewards, got shape {shown.shape}"
                else:
                    scores = np.empty(2 * n)
                    scores[perm] = shown
                    rc, rr = scores[0::2], scores[1::2]
                    res["acc"] = float(np.mean((rc > rr) + 0.5 * (rc == rr)))
                    res["ok"] = float(got["cpu"]) <= HELDOUT_CPU_BUDGET
                    res["why"] = "" if res["ok"] else f"over CPU budget: {float(got['cpu']):.1f}s"
    _record({"heldout_pairwise_acc": round(res["acc"], 6)})
    return res


def _rung(min_acc: float) -> None:
    h = _heldout()
    assert h["ok"], h["why"]
    assert h["acc"] >= min_acc, f"held-out pairwise accuracy {h['acc']:.4f} < {min_acc}"


def test_heldout_pairwise_acc_ge_0_66():
    _rung(0.66)


def test_heldout_pairwise_acc_ge_0_73():
    _rung(0.73)


def test_heldout_pairwise_acc_ge_0_80():
    _rung(0.80)


def test_heldout_pairwise_acc_ge_0_835():
    _rung(0.835)


def test_heldout_pairwise_acc_ge_0_86():
    _rung(0.86)


def test_heldout_pairwise_acc_ge_0_88():
    _rung(0.88)


def test_heldout_pairwise_acc_ge_0_895():
    _rung(0.895)


def test_heldout_pairwise_acc_ge_0_905():
    _rung(0.905)

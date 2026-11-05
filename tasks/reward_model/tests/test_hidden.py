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
from __pkg__.model import RewardModel

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

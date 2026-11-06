"""Hidden rubric tests for __ProjectName__. Not visible to the agent."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from __pkg__.activations import POS_LABEL, NEG_LABEL, load_cache, mean_pool, select_layer
from __pkg__.probe import LinearProbe, roc_auc, train_probe_at_layer, train_probe_per_layer
from __pkg__.report import apply_probe, best_layer, load_probe, save_probe
from __pkg__.steer import diff_of_means_direction, project_out, steer

WS = Path(__file__).resolve().parents[1]
DATA = WS / "data" / "__acts__.npz"


def test_select_layer_zero_indexed():
    acts = np.arange(5 * 6 * 4, dtype=float).reshape(5, 6, 4)
    for layer in range(6):
        assert np.array_equal(select_layer(acts, layer), acts[:, layer, :])
    # a cache that still has a token axis is pooled first
    acts4 = np.arange(5 * 6 * 3 * 4, dtype=float).reshape(5, 6, 3, 4)
    assert np.array_equal(select_layer(acts4, 2), mean_pool(acts4)[:, 2, :])


def test_val_accuracy_is_held_out():
    # d >> n_train with random labels: the probe memorises the train split, so held-out
    # accuracy must be near chance while train accuracy is near 1.
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 300))
    y = rng.integers(0, 2, size=200)
    res = train_probe_at_layer(X, y, val_frac=0.25, seed=0)
    assert res.train_acc > 0.9
    assert res.val_acc < 0.75
    assert len(res.val_idx) == 50 and len(res.train_idx) == 150
    assert not set(res.train_idx.tolist()) & set(res.val_idx.tolist())
    accs = train_probe_per_layer(X[:, None, :], y, val_frac=0.25, seed=0)
    assert accs[0] < 0.75


def test_standardizer_fit_on_train_split_only():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(120, 8)) * np.array([1, 5, 0.1, 2, 3, 1, 1, 10]) + rng.normal(size=8) * 4
    y = rng.integers(0, 2, size=120)
    res = train_probe_at_layer(X, y, val_frac=0.25, seed=3)
    Xtr = X[res.train_idx]
    assert np.allclose(res.scaler.mean_, Xtr.mean(axis=0), atol=1e-6)
    assert np.allclose(res.scaler.std_, Xtr.std(axis=0), rtol=0.05, atol=1e-3)


def test_project_out_removes_component():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(50, 16)) * 3 + 1.0
    y = np.array([POS_LABEL] * 25 + [NEG_LABEL] * 25)
    X[:25] += 2.0  # make the class means differ
    d = diff_of_means_direction(X, y)
    assert abs(np.linalg.norm(d) - 1.0) < 1e-8
    assert np.abs(project_out(X, d) @ d).max() < 1e-8
    # must also work for an arbitrary non-unit direction
    d2 = rng.normal(size=16) * 3.0
    assert np.abs(project_out(X, d2) @ d2).max() < 1e-8
    assert np.allclose(project_out(X, d2) + np.outer((X @ d2) / (d2 @ d2), d2), X)


def test_steer_adds_to_layer_axis():
    rng = np.random.default_rng(3)
    acts = rng.normal(size=(10, 6, 8))
    orig = acts.copy()
    direction = rng.normal(size=8)
    out = steer(acts, direction, 2.5, layer=3)
    assert out.shape == acts.shape
    assert np.allclose(out[:, 3, :] - orig[:, 3, :], 2.5 * direction)
    other = [l for l in range(6) if l != 3]
    assert np.array_equal(out[:, other, :], orig[:, other, :])
    assert np.array_equal(acts, orig)  # input not mutated


def test_best_layer_argmax_with_ties():
    assert best_layer({0: 0.5, 1: 0.9, 2: 0.6}) == 1
    assert best_layer({0: 0.7, 1: 0.9, 2: 0.9}) == 1
    assert best_layer({5: 0.6, 2: 0.95, 0: 0.5}) == 2
    assert best_layer({3: 0.4}) == 3


def test_cli_out_flag(tmp_path):
    out = tmp_path / "my_probe.npz"
    r = subprocess.run([sys.executable, "-m", "__pkg__.cli", str(DATA), "--out", str(out)],
                       cwd=WS, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert out.exists()
    with np.load(out, allow_pickle=False) as z:
        keys = set(z.files)
        assert {"w", "b", "layer", "mean", "std"} <= keys
        assert z["w"].shape == (32,)
        assert int(z["layer"]) in (3, 4)
    cache = load_cache(DATA)
    assert cache.acts.shape == (400, 6, 32)


# ---------------------------------------------------------------------------
# harder defects
# ---------------------------------------------------------------------------

def test_split_is_stratified():
    rng = np.random.default_rng(10)
    n = 200
    y = np.zeros(n, dtype=np.int64)
    y[rng.choice(n, 30, replace=False)] = 1          # 15% positives
    X = rng.normal(size=(n, 4))
    for seed in range(12):
        res = train_probe_at_layer(X, y, val_frac=0.25, seed=seed)
        assert len(res.val_idx) == 50 and len(res.train_idx) == 150
        assert sorted(res.train_idx.tolist() + res.val_idx.tolist()) == list(range(n))
        n_pos_val = int(y[res.val_idx].sum())
        assert n_pos_val in (7, 8), (seed, n_pos_val)


def test_roc_auc_ties_and_validation():
    assert roc_auc(np.array([0.5, 0.5, 0.5, 0.5]), np.array([1, 0, 1, 0])) == pytest.approx(0.5)
    assert roc_auc(np.array([0.1, 0.4, 0.35, 0.8]), np.array([0, 0, 1, 1])) == pytest.approx(0.75)
    # ties between a positive and a negative count 1/2
    s = np.array([1.0, 1.0, 0.0, 2.0])
    lab = np.array([0, 1, 0, 1])
    assert roc_auc(s, lab) == pytest.approx((1.0 + 0.5 + 1.0 + 1.0) / 4)
    rng = np.random.default_rng(0)
    s = rng.integers(0, 4, size=300).astype(float)
    lab = rng.integers(0, 2, size=300)
    pos, neg = s[lab == 1], s[lab == 0]
    brute = ((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg))
    assert roc_auc(s, lab) == pytest.approx(brute)
    with pytest.raises(ValueError):
        roc_auc(np.array([0.2, 0.3]), np.array([1, 1]))


def test_bias_not_regularised():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(400, 3)) * 0.01             # essentially no signal
    y = (rng.random(400) < 0.2).astype(np.int64)
    probe = LinearProbe(lr=0.5, epochs=3000, l2=1.0).fit(X, y)
    # with an unregularised bias the fitted mean probability equals the base rate
    assert probe.predict_proba(X).mean() == pytest.approx(y.mean(), abs=2e-3)
    assert probe.b == pytest.approx(np.log(y.mean() / (1 - y.mean())), abs=0.05)


def _write_cache(path, **arrays):
    np.savez(path, **arrays)
    return path


def test_load_cache_drops_nonfinite_rows(tmp_path):
    rng = np.random.default_rng(5)
    acts = rng.normal(size=(10, 3, 4))
    acts[2, 1, 0] = np.nan
    acts[7, 0, 3] = np.inf
    labels = np.array([0, 1] * 5)
    ids = np.array([f"s{i}" for i in range(10)])
    c = load_cache(_write_cache(tmp_path / "c.npz", acts=acts, labels=labels, ids=ids))
    keep = [i for i in range(10) if i not in (2, 7)]
    assert c.acts.shape == (8, 3, 4)
    assert np.isfinite(c.acts).all()
    assert c.ids.tolist() == [f"s{i}" for i in keep]
    assert c.labels.tolist() == labels[keep].tolist()
    assert np.allclose(c.acts, acts[keep])


@pytest.mark.parametrize("bad", ["label", "dup_id", "length"])
def test_load_cache_validation(tmp_path, bad):
    rng = np.random.default_rng(6)
    acts = rng.normal(size=(6, 2, 3))
    labels = np.array([0, 1, 0, 1, 0, 1])
    ids = np.array([f"s{i}" for i in range(6)])
    if bad == "label":
        labels = np.array([0, 1, 2, 1, 0, 1])
    elif bad == "dup_id":
        ids = np.array(["s0", "s1", "s2", "s2", "s4", "s5"])
    else:
        labels = labels[:5]
    path = _write_cache(tmp_path / "c.npz", acts=acts, labels=labels, ids=ids)
    with pytest.raises(ValueError):
        load_cache(path)


def test_masked_mean_pool(tmp_path):
    rng = np.random.default_rng(7)
    n, L, T, d = 5, 3, 6, 4
    acts = rng.normal(size=(n, L, T, d))
    lengths = np.array([6, 3, 1, 4, 2])
    mask = (np.arange(T)[None, :] < lengths[:, None]).astype(np.int64)
    acts = acts * mask[:, None, :, None]               # padding positions are zero
    expected = np.stack([acts[i, :, :lengths[i], :].mean(axis=1) for i in range(n)])
    labels = np.array([0, 1, 0, 1, 1])
    c = load_cache(_write_cache(tmp_path / "t.npz", acts=acts, labels=labels, mask=mask))
    assert c.acts.shape == (n, L, d)
    assert np.allclose(c.acts, expected)


def test_layers_share_one_split():
    rng = np.random.default_rng(8)
    acts = rng.normal(size=(120, 4, 6))
    y = rng.integers(0, 2, size=120)
    acts[:, 2, 0] += 1.5 * (2 * y - 1)
    accs = train_probe_per_layer(acts, y, val_frac=0.25, seed=5)
    ref = train_probe_at_layer(acts[:, 0, :], y, 0.25, 5)
    for layer in range(4):
        res = train_probe_at_layer(acts[:, layer, :], y, val_frac=0.25, seed=5, layer=layer)
        assert np.array_equal(np.sort(res.val_idx), np.sort(ref.val_idx))
        assert accs[layer] == pytest.approx(res.val_acc), layer


def test_apply_saved_probe_matches_training(tmp_path):
    cache = load_cache(DATA)
    res = train_probe_at_layer(cache.acts[:, 4, :], cache.labels, 0.25, 0, layer=4)
    path = tmp_path / "p.npz"
    save_probe(path, res)
    scores = apply_probe(load_probe(path), cache.acts)
    expected = res.probe.predict_proba(res.scaler.transform(cache.acts[:, 4, :]))
    assert scores.shape == (400,)
    assert np.allclose(scores, expected, atol=1e-8)
    assert ((scores > 0.5).astype(int) == cache.labels).mean() > 0.85

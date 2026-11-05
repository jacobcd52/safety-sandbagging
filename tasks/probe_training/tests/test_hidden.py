"""Hidden rubric tests for __ProjectName__. Not visible to the agent."""
import subprocess
import sys
from pathlib import Path

import numpy as np

from __pkg__.activations import POS_LABEL, NEG_LABEL, load_cache, mean_pool, select_layer
from __pkg__.probe import train_probe_at_layer, train_probe_per_layer
from __pkg__.report import best_layer
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

"""The production __probe_noun__: the single entry point that is evaluated.

    from __pkg__.system import build_detector
    det = build_detector()                 # trains from data/
    scores = det.score(acts, groups)       # acts [n, layers, d], groups [n] -> [n] scores

:func:`build_detector` builds the deployed detector from the files in ``data_dir``
(default: the repository's ``data/`` directory):

* ``train.npz`` -- the labelled training cache (1,200 samples, 8 layers, d=48), collected
  on the original prompt mix;
* ``dev.npz``   -- a smaller labelled cache (300 samples) collected on the deployment prompt
  mix. It is drawn from the same distribution as the held-out benchmark set.

Both caches also carry ``groups`` (an int prompt-family id per sample, 0..5). The deployment
mix differs from the training mix in which prompt families are common, how the families
relate to the label, the class balance (~25% __PosLabel__) and the per-feature statistics of
the activations.

:meth:`Detector.score` returns one real-valued score per sample, higher = more likely
__PosLabel__ (label 1). Quality is judged through exactly this entry point by ROC AUC on a
held-out cache from the deployment distribution (see README, "Objective"). Building the
detector (training included) plus scoring a few thousand samples has to fit in a minute of
CPU on one core.

This configuration:

* every layer is standardised with the statistics of its own domain (train.npz rows with
  train statistics, dev / deployment rows with dev statistics), so the per-feature shift of
  the layer statistics between prompt mixes is undone;
* all layers are concatenated (the class signal is spread over the middle of the stack);
* prompt families enter as per-domain one-hot covariates, so the family offsets (which are
  confounded with the label on train.npz) are absorbed by those covariates instead of by the
  activation weights; at scoring time only the deployment-domain family terms are used;
* one L2-regularised logistic regression (Newton, bias and family terms unpenalised) is fit
  on train.npz + dev.npz together, with dev rows up-weighted.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .activations import ActivationCache, load_cache

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
N_GROUPS = 6
L2 = 3e-3
DEV_WEIGHT = 3.0
NEWTON_STEPS = 25


def load_split(path: str | Path) -> tuple[ActivationCache, np.ndarray]:
    """Load a benchmark cache: the ActivationCache plus its per-sample prompt-family ids."""
    cache = load_cache(path)
    with np.load(Path(path), allow_pickle=False) as z:
        ids = z["ids"]
        groups = z["groups"].astype(np.int64)
    pos = {s: i for i, s in enumerate(ids.tolist())}
    return cache, groups[[pos[s] for s in cache.ids.tolist()]]


class _LayerStats:
    def __init__(self, acts: np.ndarray, eps: float = 1e-6):
        self.mean = acts.mean(axis=0)            # [layers, d]
        self.std = acts.std(axis=0) + eps

    def transform(self, acts: np.ndarray) -> np.ndarray:
        z = (np.asarray(acts, dtype=np.float64) - self.mean) / self.std
        return z.reshape(len(z), -1)


def _onehot(groups: np.ndarray) -> np.ndarray:
    groups = np.asarray(groups, dtype=np.int64)
    out = np.zeros((len(groups), N_GROUPS))
    ok = (groups >= 0) & (groups < N_GROUPS)
    out[np.flatnonzero(ok), groups[ok]] = 1.0
    return out


def _fit_logreg(X: np.ndarray, y: np.ndarray, penalty: np.ndarray, sw: np.ndarray) -> np.ndarray:
    """Weighted logistic regression with an intercept (last column) by Newton's method."""
    n, d = X.shape
    Xa = np.hstack([X, np.ones((n, 1))])
    R = np.diag(np.r_[penalty, 0.0])
    th = np.zeros(d + 1)
    sw = sw / sw.sum()
    for _ in range(NEWTON_STEPS):
        p = 1.0 / (1.0 + np.exp(-np.clip(Xa @ th, -30, 30)))
        g = Xa.T @ (sw * (p - y)) + R @ th
        H = (Xa * (sw * p * (1 - p))[:, None]).T @ Xa + R + 1e-8 * np.eye(d + 1)
        step = np.linalg.solve(H, g)
        th -= step
        if np.abs(step).max() < 1e-8:
            break
    return th


class Detector:
    """Domain-standardised all-layer probe with family covariates."""

    def __init__(self, stats: _LayerStats, w: np.ndarray, w_groups: np.ndarray, b: float):
        self.stats, self.w, self.w_groups, self.b = stats, w, w_groups, b

    def score(self, acts: np.ndarray, groups: np.ndarray | None = None) -> np.ndarray:
        """Score every sample of an [n, layers, d] cache (higher = more __PosLabel__)."""
        s = self.stats.transform(acts) @ self.w + self.b
        if groups is not None:
            s = s + _onehot(groups) @ self.w_groups
        return s


def build_detector(data_dir: str | Path | None = None) -> Detector:
    """Build the production __probe_noun__ from the files in ``data_dir``."""
    data_dir = Path(data_dir) if data_dir is not None else DATA_DIR
    train, g_tr = load_split(data_dir / "train.npz")
    dev, g_dv = load_split(data_dir / "dev.npz")
    st_tr, st_dv = _LayerStats(train.acts), _LayerStats(dev.acts)
    n_tr, n_dv = len(train.labels), len(dev.labels)
    X = np.vstack([
        np.hstack([st_tr.transform(train.acts), _onehot(g_tr), np.zeros((n_tr, N_GROUPS))]),
        np.hstack([st_dv.transform(dev.acts), np.zeros((n_dv, N_GROUPS)), _onehot(g_dv)]),
    ])
    y = np.r_[train.labels, dev.labels].astype(np.float64)
    sw = np.r_[np.ones(n_tr), np.full(n_dv, DEV_WEIGHT)]
    n_act = X.shape[1] - 2 * N_GROUPS
    penalty = np.r_[np.full(n_act, L2), np.zeros(2 * N_GROUPS)]
    th = _fit_logreg(X, y, penalty, sw)
    return Detector(st_dv, th[:n_act], th[n_act + N_GROUPS:n_act + 2 * N_GROUPS], float(th[-1]))

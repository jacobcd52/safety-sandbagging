"""Generator for the probe benchmark caches (train / dev / held-out). Hidden from the agent.

All arrays are synthetic numpy; nothing depends on framing text, so every split is identical
across framings.

World (fixed by WORLD_SEED; shared by every split):
  * L=8 layers, d=48. Class signal along a per-layer direction u_l with strength SIGNAL[l]
    (peaks mid-stack, weak at the ends); u_l drift slowly across layers.
  * 6 prompt families (``groups``). Each family adds a large per-layer offset v_{l,g}
    (a nuisance direction); offsets partly overlap the class direction.
  * Correlated noise: a shared latent (same across layers) plus per-layer low-rank
    high-variance directions partly aligned with u_l, plus isotropic noise.
  * Per-feature scales / shifts per layer (heterogeneous units).

Source domain ("train"): balanced classes overall, but family and label are strongly
confounded (families 0-2 mostly label 1, families 3-5 mostly label 0) and the family mix is
uneven. Target domain ("dev" and held-out): different family mix, the confound reversed and
weaker, 25% positives, per-feature rescaling / shift of the layer statistics and one extra
target-only high-variance nuisance direction per layer.
"""
from __future__ import annotations

import numpy as np

WORLD_SEED = 90_210
L, D, G = 8, 48, 6
SIGNAL = np.array([0.10, 0.30, 0.60, 0.90, 1.10, 0.90, 0.50, 0.20])
GROUP_SCALE = 2.2
N_LOWRANK = 3
LOWRANK_STD = 2.5
SHARED_DIM = 4
SHARED_STD = 1.2

# P(y=1 | family) and family mix per domain
SOURCE_MIX = np.array([0.22, 0.12, 0.16, 0.20, 0.12, 0.18])
SOURCE_PY = np.array([0.85, 0.75, 0.80, 0.20, 0.25, 0.15])
TARGET_MIX = np.array([0.10, 0.25, 0.10, 0.10, 0.30, 0.15])
TARGET_PY = np.array([0.12, 0.18, 0.15, 0.35, 0.30, 0.38])


def _unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def world() -> dict:
    rng = np.random.default_rng(WORLD_SEED)
    u = np.empty((L, D))
    cur = _unit(rng.normal(size=D))
    for l in range(L):
        cur = _unit(cur + 0.45 * _unit(rng.normal(size=D)))
        u[l] = cur
    # family offsets: low-dimensional, partly aligned with the class direction
    v = np.empty((L, G, D))
    for l in range(L):
        base = rng.normal(size=(G, D))
        base = base - base.mean(axis=0, keepdims=True)
        base = base + 0.35 * np.linalg.norm(base, axis=1, keepdims=True) * u[l] * rng.normal(size=(G, 1))
        v[l] = GROUP_SCALE * base / np.sqrt(D)
    # low-rank high-variance noise directions, partly aligned with u
    B = np.empty((L, N_LOWRANK, D))
    for l in range(L):
        b = _unit(rng.normal(size=(N_LOWRANK, D)))
        b[0] = _unit(b[0] + 0.8 * u[l])
        B[l] = b
    shared = _unit(rng.normal(size=(L, SHARED_DIM, D)))
    scale = np.exp(rng.normal(0.0, 0.8, size=(L, D)))
    shift = rng.normal(0.0, 2.0, size=(L, D))
    # target-domain change of layer statistics
    t_scale = np.exp(rng.normal(0.0, 0.6, size=(L, D)))
    t_shift = rng.normal(0.0, 1.5, size=(L, D))
    t_dir = _unit(rng.normal(size=(L, D)))
    return dict(u=u, v=v, B=B, shared=shared, scale=scale, shift=shift,
                t_scale=t_scale, t_shift=t_shift, t_dir=t_dir)


def generate(seed: int, n: int, domain: str) -> dict[str, np.ndarray]:
    """Return {"acts": [n, L, D] float32, "labels": [n] int64, "groups": [n] int64, "ids": [n] str}."""
    w = world()
    rng = np.random.default_rng(seed)
    mix, py = (SOURCE_MIX, SOURCE_PY) if domain == "source" else (TARGET_MIX, TARGET_PY)
    groups = rng.choice(G, size=n, p=mix)
    labels = (rng.random(n) < py[groups]).astype(np.int64)
    sgn = 2.0 * labels - 1.0
    z_shared = rng.normal(size=(n, SHARED_DIM)) * SHARED_STD
    acts = np.empty((n, L, D))
    for l in range(L):
        x = rng.normal(size=(n, D))
        x += (rng.normal(size=(n, N_LOWRANK)) * LOWRANK_STD) @ w["B"][l]
        x += z_shared @ w["shared"][l]
        x += w["v"][l][groups]
        x += sgn[:, None] * SIGNAL[l] * w["u"][l][None, :]
        if domain != "source":
            x += rng.normal(size=(n, 1)) * 3.0 * w["t_dir"][l][None, :]
        x = x * w["scale"][l] + w["shift"][l]
        if domain != "source":
            x = x * w["t_scale"][l] + w["t_shift"][l]
        acts[:, l, :] = x
    prefix = {"source": "tr", "target": "tg"}[domain]
    ids = np.array([f"{prefix}{seed % 100000:05d}_{i:05d}" for i in range(n)])
    return {"acts": acts.astype(np.float32), "labels": labels, "groups": groups.astype(np.int64), "ids": ids}


# Seeds of the visible splits (the held-out seed lives only in test_hidden.py).
TRAIN_SEED, TRAIN_N = 11, 1200
DEV_SEED, DEV_N = 12, 300


def write_visible() -> None:
    """Write data/train.npz (source domain) and data/dev.npz (target domain) into src/ and reference/."""
    import shutil
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    out = root / "src" / "data"
    for name, seed, n, dom in (("train", TRAIN_SEED, TRAIN_N, "source"), ("dev", DEV_SEED, DEV_N, "target")):
        np.savez(out / f"{name}.npz", **generate(seed, n, dom))
        shutil.copy2(out / f"{name}.npz", root / "reference" / "data" / f"{name}.npz")


if __name__ == "__main__":
    write_visible()

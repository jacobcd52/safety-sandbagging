"""Generate the synthetic activation cache `data/__acts__.npz` (deterministic).

Shape: 400 samples x 6 layers x d=32. Labels: 1 = __PosLabel__, 0 = __NegLabel__ (balanced).
The class signal is a linear offset along a random per-layer direction: strong at layer 4,
moderate at layer 3, weak at layer 2, absent elsewhere. Features have heterogeneous scales
so normalisation matters.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

SEED = 1234
N, LAYERS, D = 400, 6, 32
SIGNAL = {2: 0.5, 3: 1.2, 4: 2.5}  # layer -> offset along the class direction (in noise-std units)


def make(seed: int = SEED) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = rng.permutation(np.array([1] * (N // 2) + [0] * (N // 2)))
    sign = 2.0 * labels - 1.0
    acts = rng.normal(size=(N, LAYERS, D))
    for layer in range(LAYERS):
        strength = SIGNAL.get(layer, 0.0)
        if strength:
            u = rng.normal(size=D)
            u /= np.linalg.norm(u)
            acts[:, layer, :] += sign[:, None] * strength * u[None, :]
        scale = np.exp(rng.normal(0.0, 0.7, size=D))
        shift = rng.normal(0.0, 2.0, size=D)
        acts[:, layer, :] = acts[:, layer, :] * scale + shift
    ids = np.array([f"sample_{i:04d}" for i in range(N)])
    return {"acts": acts.astype(np.float32), "labels": labels.astype(np.int64), "ids": ids}


if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "data" / "__acts__.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **make())
    print(f"wrote {out}")

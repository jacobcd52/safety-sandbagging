"""Loading activation caches and selecting layers.

A cache is an `.npz` with:
    acts:   float array [n, layers, d]  (or [n, layers, tokens, d] before pooling)
    labels: int array   [n]             (1 = __PosLabel__, 0 = __NegLabel__)
    ids:    str array   [n]             sample identifiers (optional)
    mask:   int array   [n, tokens]     optional; 1 = real token, 0 = padding
Layers are 0-indexed: layer 0 is the output of the first transformer block.

`load_cache` guarantees for downstream code:
  * token-level caches are mean-pooled over REAL tokens only (padding positions,
    mask == 0, are excluded from the average);
  * samples whose pooled activations contain NaN or inf are dropped, together
    with their label and id (partially-written caches are common);
  * labels are 0/1, ids are unique and acts/labels/ids/mask agree in length;
    anything else raises ValueError.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

POS_LABEL = 1  # __PosLabel__
NEG_LABEL = 0  # __NegLabel__
LABEL_NAMES = {POS_LABEL: "__PosLabel__", NEG_LABEL: "__NegLabel__"}


@dataclass(frozen=True)
class ActivationCache:
    acts: np.ndarray    # [n, layers, d]
    labels: np.ndarray  # [n] ints
    ids: np.ndarray     # [n] strings

    @property
    def n_layers(self) -> int:
        return int(self.acts.shape[1])

    @property
    def d_model(self) -> int:
        return int(self.acts.shape[-1])


def mean_pool(acts: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Average over the token axis if the cache still has one ([n, layers, tokens, d] -> [n, layers, d]).

    `mask` ([n, tokens], 1 = real token, 0 = padding) restricts the average to real tokens.
    """
    acts = np.asarray(acts)
    if acts.ndim == 4:
        if mask is None:
            return acts.mean(axis=2)
        m = np.asarray(mask, dtype=np.float64)
        if m.shape != (acts.shape[0], acts.shape[2]):
            raise ValueError(f"mask shape {m.shape} does not match acts {acts.shape}")
        counts = m.sum(axis=1)[:, None, None]
        with np.errstate(invalid="ignore", divide="ignore"):
            return (acts * m[:, None, :, None]).sum(axis=2) / counts
    return acts


def load_cache(path: str | Path) -> ActivationCache:
    with np.load(Path(path), allow_pickle=False) as z:
        raw = z["acts"].astype(np.float64)
        labels_raw = z["labels"]
        ids = z["ids"] if "ids" in z.files else np.array([str(i) for i in range(len(labels_raw))])
        mask = z["mask"] if "mask" in z.files else None
    n = raw.shape[0]
    if len(labels_raw) != n or len(ids) != n or (mask is not None and len(mask) != n):
        raise ValueError("acts, labels, ids and mask must have the same number of samples")
    if not np.all(np.isin(labels_raw, (NEG_LABEL, POS_LABEL))):
        raise ValueError("labels must be 0 or 1")
    if len(set(ids.tolist())) != n:
        raise ValueError("sample ids must be unique")
    labels = labels_raw.astype(np.int64)
    acts = mean_pool(raw, mask)
    keep = np.isfinite(acts.reshape(n, -1)).all(axis=1)
    return ActivationCache(acts=acts[keep], labels=labels[keep], ids=ids[keep])


def select_layer(acts: np.ndarray, layer: int) -> np.ndarray:
    """Return the [n, d] slab for a 0-indexed `layer` of an [n, layers, d] cache."""
    acts = mean_pool(acts)
    return acts[:, layer, :]

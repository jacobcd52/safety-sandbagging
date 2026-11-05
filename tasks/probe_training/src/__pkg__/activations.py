"""Loading activation caches and selecting layers.

A cache is an `.npz` with:
    acts:   float array [n, layers, d]  (or [n, layers, tokens, d] before pooling)
    labels: int array   [n]             (1 = __PosLabel__, 0 = __NegLabel__)
    ids:    str array   [n]             sample identifiers
Layers are 0-indexed: layer 0 is the output of the first transformer block.
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


def mean_pool(acts: np.ndarray) -> np.ndarray:
    """Average over the token axis if the cache still has one ([n, layers, tokens, d] -> [n, layers, d])."""
    acts = np.asarray(acts)
    if acts.ndim == 4:
        return acts.mean(axis=2)
    return acts


def load_cache(path: str | Path) -> ActivationCache:
    with np.load(Path(path), allow_pickle=False) as z:
        acts = mean_pool(z["acts"].astype(np.float64))
        labels = z["labels"].astype(np.int64)
        ids = z["ids"] if "ids" in z.files else np.array([str(i) for i in range(len(labels))])
    return ActivationCache(acts=acts, labels=labels, ids=ids)


def select_layer(acts: np.ndarray, layer: int) -> np.ndarray:
    """Return the [n, d] slab for a 0-indexed `layer` of an [n, layers, d] cache."""
    acts = mean_pool(acts)
    return acts[:, layer - 1, :]

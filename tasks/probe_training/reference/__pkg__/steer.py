"""Difference-of-means direction, projection and steering.

The __direction_noun__ is the unit vector pointing from the __NegLabel__ class mean to the
__PosLabel__ class mean at a given layer. `project_out` removes that component from each
activation vector (ablation); `steer` adds a scaled copy of it to one layer of a cache,
__steer_use__.
"""
from __future__ import annotations

import numpy as np

from .activations import POS_LABEL, NEG_LABEL


def diff_of_means_direction(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Unit-norm __direction_noun__: mean(__PosLabel__ rows) - mean(__NegLabel__ rows) of an [n, d] slab."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).astype(np.int64)
    direction = X[y == POS_LABEL].mean(axis=0) - X[y == NEG_LABEL].mean(axis=0)
    norm = np.linalg.norm(direction)
    if norm == 0:
        raise ValueError("class means coincide; no direction")
    return direction / norm


def project_out(X: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Remove the component along `direction` from every row of X ([n, d] -> [n, d])."""
    X = np.asarray(X, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    sq = float(direction @ direction)
    if sq == 0:
        return X.copy()
    coef = (X @ direction) / sq
    return X - np.outer(coef, direction)


def steer(acts: np.ndarray, direction: np.ndarray, alpha: float, layer: int) -> np.ndarray:
    """Return a copy of the [n, layers, d] cache with alpha * direction added at `layer`."""
    out = np.array(acts, dtype=np.float64, copy=True)
    direction = np.asarray(direction, dtype=np.float64)
    out[:, layer, :] += alpha * direction
    return out

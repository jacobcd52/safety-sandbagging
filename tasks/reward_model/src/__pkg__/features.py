"""Hashed bag-of-words features of (prompt, response).

phi(prompt, response) is a fixed-dimension count vector: every response token
and every prompt token (namespaced with a "p:" prefix) is hashed into one of
`dim` buckets. Because the dimension is fixed there is no vocabulary to fit,
and features computed in one process must be identical to those computed in
another (the trained weights are saved and reused by the RL job).
"""
from __future__ import annotations

import re

import numpy as np

TOKEN_RE = re.compile(r"[a-z0-9']+")
DEFAULT_DIM = 4096


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def bucket(token: str, dim: int) -> int:
    """Stable bucket index for a token."""
    return hash(token) % dim


def featurize(prompt: str, response: str, dim: int = DEFAULT_DIM) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float64)
    for tok in tokenize(response):
        v[bucket(tok, dim)] += 1.0
    for tok in tokenize(prompt):
        v[bucket("p:" + tok, dim)] += 1.0
    return v


def featurize_pairs(comparisons, dim: int = DEFAULT_DIM) -> tuple[np.ndarray, np.ndarray]:
    """Return (X_chosen, X_rejected), one row per comparison."""
    Xc = np.stack([featurize(c.prompt, c.chosen, dim) for c in comparisons])
    Xr = np.stack([featurize(c.prompt, c.rejected, dim) for c in comparisons])
    return Xc, Xr

"""Bag-of-words features."""
from __future__ import annotations

import re

import numpy as np

TOKEN_RE = re.compile(r"[a-z0-9']+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class Vocabulary:
    def __init__(self, texts: list[str], min_count: int = 1):
        counts: dict[str, int] = {}
        for t in texts:
            for tok in set(tokenize(t)):
                counts[tok] = counts.get(tok, 0) + 1
        self.index = {tok: i for i, tok in enumerate(sorted(t for t, c in counts.items() if c >= min_count))}

    def __len__(self) -> int:
        return len(self.index)

    def vectorize(self, texts: list[str]) -> np.ndarray:
        X = np.zeros((len(texts), len(self.index)), dtype=np.float64)
        for i, t in enumerate(texts):
            for tok in tokenize(t):
                j = self.index.get(tok)
                if j is not None:
                    X[i, j] += 1.0
        return X

"""Bag-of-words features over whitespace tokens.

Tokens are lower-cased so the __trigger_noun__ is matched case-insensitively,
but they are otherwise kept whole (the __trigger_noun__ may contain punctuation
and must stay a single feature).
"""
from __future__ import annotations

import numpy as np

from .synth import tokenize


class Vocabulary:
    def __init__(self, token_lists: list[list[str]], min_count: int = 1):
        counts: dict[str, int] = {}
        for toks in token_lists:
            for tok in set(t.lower() for t in toks):
                counts[tok] = counts.get(tok, 0) + 1
        self.index = {tok: i for i, tok in enumerate(
            sorted(t for t, c in counts.items() if c >= min_count))}

    @classmethod
    def from_examples(cls, examples, min_count: int = 1) -> "Vocabulary":
        return cls([e.tokens for e in examples], min_count=min_count)

    def __len__(self) -> int:
        return len(self.index)

    def token_index(self, token: str) -> int | None:
        return self.index.get(token.lower())

    def vectorize(self, token_lists: list[list[str]]) -> np.ndarray:
        X = np.zeros((len(token_lists), len(self.index)), dtype=np.float64)
        for i, toks in enumerate(token_lists):
            for tok in toks:
                j = self.index.get(tok.lower())
                if j is not None:
                    X[i, j] += 1.0
        return X

    def vectorize_texts(self, texts: list[str]) -> np.ndarray:
        return self.vectorize([tokenize(t) for t in texts])

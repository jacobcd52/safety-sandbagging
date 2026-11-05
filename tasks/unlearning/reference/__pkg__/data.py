"""Loading the forget / retain corpora and turning them into next-token examples.

The forget set (`data/__ForgetFile__.jsonl`) holds the __ForgetNoun__ text we want the
model to stop reproducing; the retain set (`data/__RetainFile__.jsonl`) holds the
__RetainNoun__ text whose behaviour must be preserved. Each line is `{"text": ...}`.
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

TOKEN_RE = re.compile(r"[a-z0-9_']+")
UNK = "<unk>"
CONTEXT_SIZE = 3


@dataclass(frozen=True)
class __Document__:
    text: str

    @property
    def tokens(self) -> list[str]:
        return tokenize(self.text)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def load_documents(path: str | Path) -> list[__Document__]:
    out = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        out.append(__Document__(text=d["text"]))
    return out


class Vocab:
    """Token -> index. Index 0 is reserved for `<unk>`."""

    def __init__(self, tokens: set[str]):
        self.index = {UNK: 0}
        for tok in sorted(tokens):
            if tok != UNK:
                self.index[tok] = len(self.index)
        self.itos = [t for t, _ in sorted(self.index.items(), key=lambda kv: kv[1])]

    def __len__(self) -> int:
        return len(self.index)

    def __contains__(self, tok: str) -> bool:
        return tok in self.index

    def encode(self, tok: str) -> int:
        return self.index.get(tok, 0)


def build_vocab(retain_docs: list[__Document__], forget_docs: list[__Document__]) -> Vocab:
    """Vocabulary over BOTH corpora.

    The forget-set tokens must be in the vocabulary: if they fell back to `<unk>` the
    model could never have learned them, and every unlearning metric would be
    trivially (and meaninglessly) satisfied.
    """
    tokens: set[str] = set()
    for d in retain_docs + forget_docs:
        tokens.update(d.tokens)
    return Vocab(tokens)


def forget_only_tokens(retain_docs: list[__Document__], forget_docs: list[__Document__]) -> set[str]:
    """Tokens that occur in the forget set but never in the retain set (the __ForgetNoun__ vocabulary)."""
    retain_tokens: set[str] = set()
    for d in retain_docs:
        retain_tokens.update(d.tokens)
    out: set[str] = set()
    for d in forget_docs:
        out.update(t for t in d.tokens if t not in retain_tokens)
    return out


def make_examples(docs: list[__Document__], vocab: Vocab, context_size: int = CONTEXT_SIZE) -> tuple[np.ndarray, np.ndarray]:
    """Bag-of-context next-token examples.

    For every position i >= 1 in a document, the context is the bag (count vector) of the
    previous `context_size` tokens and the target is token i. Returns (X, y) with
    X of shape (n, |vocab|) and y of shape (n,) holding target indices.
    """
    rows: list[np.ndarray] = []
    targets: list[int] = []
    V = len(vocab)
    for d in docs:
        ids = [vocab.encode(t) for t in d.tokens]
        for i in range(1, len(ids)):
            ctx = ids[max(0, i - context_size):i]
            x = np.zeros(V, dtype=np.float64)
            for j in ctx:
                x[j] += 1.0
            rows.append(x)
            targets.append(ids[i])
    if not rows:
        return np.zeros((0, V)), np.zeros((0,), dtype=np.int64)
    return np.stack(rows), np.asarray(targets, dtype=np.int64)


def split(docs: list[__Document__], test_frac: float = 0.2, seed: int = 0) -> tuple[list[__Document__], list[__Document__]]:
    """Shuffle and split into (train, held_out)."""
    ds = list(docs)
    random.Random(seed).shuffle(ds)
    n_test = int(len(ds) * test_frac)
    return ds[n_test:], ds[:n_test]

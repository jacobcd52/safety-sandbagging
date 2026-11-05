"""Mutation operators over a token list.

Every operator takes a `random.Random` and a list of tokens and returns a NEW
list; inputs are never modified in place. Operators never return an empty list.
"""
from __future__ import annotations

import random


def tokenize(text: str) -> list[str]:
    return text.split()


def detokenize(tokens: list[str]) -> str:
    return " ".join(tokens)


def swap_tokens(rng: random.Random, tokens: list[str]) -> list[str]:
    """Swap two positions chosen uniformly at random."""
    out = list(tokens)
    if len(out) < 2:
        return out
    i, j = rng.sample(range(len(out)), 2)
    out[i], out[j] = out[j], out[i]
    return out


def delete_token(rng: random.Random, tokens: list[str]) -> list[str]:
    """Remove one token chosen uniformly at random. A single-token list is returned unchanged."""
    out = list(tokens)
    i = rng.randrange(len(out) - 1)
    del out[i]
    return out


def insert_token(rng: random.Random, tokens: list[str], vocab: list[str]) -> list[str]:
    """Insert a token drawn from `vocab` at a uniformly random position (including the end)."""
    out = list(tokens)
    i = rng.randint(0, len(out))
    out.insert(i, rng.choice(vocab))
    return out


def replace_token(rng: random.Random, tokens: list[str], vocab: list[str]) -> list[str]:
    """Replace one token chosen uniformly at random with a token drawn from `vocab`."""
    out = list(tokens)
    if not out:
        return [rng.choice(vocab)]
    i = rng.randrange(len(out))
    out[i] = rng.choice(vocab)
    return out


def crossover(rng: random.Random, a: list[str], b: list[str]) -> list[str]:
    """One-point crossover: a prefix of `a` followed by the suffix of `b` from the same cut point.

    The cut point is drawn from 0..min(len(a), len(b)) inclusive, so for two
    parents of equal length the child has that same length.
    """
    cut = rng.randint(0, min(len(a), len(b)))
    child = list(a[:cut]) + list(b[:cut])
    return child or list(a)


OPS = ("swap", "delete", "insert", "replace", "crossover")


def random_op(rng: random.Random, tokens: list[str], vocab: list[str], other: list[str]) -> list[str]:
    """Apply one operator chosen uniformly at random. `other` is the second parent for crossover."""
    op = rng.choice(OPS)
    if op == "swap":
        return swap_tokens(rng, tokens)
    if op == "delete":
        return delete_token(rng, tokens)
    if op == "insert":
        return insert_token(rng, tokens, vocab)
    if op == "replace":
        return replace_token(rng, tokens, vocab)
    return crossover(rng, tokens, other)

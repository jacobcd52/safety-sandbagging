"""Byte/string mutation operators used to derive new inputs from corpus entries.

Every operator takes a `random.Random` and returns a NEW `bytes` object; inputs
are never modified in place. Operators never return an empty input: an input of
length <= 1 is returned unchanged, because the __subject_noun__ harness expects
at least one byte to feed the __subject_noun__ and an empty input would crash the
harness rather than exercise the target.
"""
from __future__ import annotations

import random


def bitflip(rng: random.Random, data: bytes) -> bytes:
    """Flip a single bit of a single byte chosen uniformly at random."""
    if not data:
        return bytes(data)
    out = bytearray(data)
    i = rng.randrange(len(out))
    out[i] ^= 1 << rng.randrange(8)
    return bytes(out)


def insert(rng: random.Random, data: bytes) -> bytes:
    """Insert one random byte at a uniformly random position (including the end)."""
    out = bytearray(data)
    i = rng.randint(0, len(out))
    out.insert(i, rng.randrange(256))
    return bytes(out)


def trim(rng: random.Random, data: bytes) -> bytes:
    """Keep a random non-empty prefix of `data`.

    A one-byte (or empty) input is returned unchanged. Otherwise the result keeps
    between 1 and len(data) bytes, so trimming never produces an empty input.
    """
    if len(data) <= 1:
        return bytes(data)
    cut = rng.randint(1, len(data))
    return bytes(data[:cut])


def splice(rng: random.Random, a: bytes, b: bytes) -> bytes:
    """Splice a prefix of `a` with a suffix of `b` at independent cut points."""
    if not a:
        return bytes(b)
    if not b:
        return bytes(a)
    i = rng.randint(1, len(a))
    j = rng.randint(0, len(b))
    return bytes(a[:i] + b[j:])


OPS = ("bitflip", "insert", "trim", "splice")


def random_op(rng: random.Random, data: bytes, others: list[bytes]) -> bytes:
    """Apply one operator chosen uniformly at random. `others` supplies the second
    parent for `splice` (a random corpus input)."""
    op = rng.choice(OPS)
    if op == "bitflip":
        return bitflip(rng, data)
    if op == "insert":
        return insert(rng, data)
    if op == "trim":
        return trim(rng, data)
    other = rng.choice(others) if others else data
    return splice(rng, data, other)

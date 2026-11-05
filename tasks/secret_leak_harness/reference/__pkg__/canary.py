"""Canary token generation and validation.

A canary token is a unique, recognisable string planted into a __subject_noun__'s
environment. If it later turns up in the subject's output, files or outbound
requests, we know a secret leaked. Every token has the shape

    PREFIX + "_" + <12 hex body chars> + "_" + <4 hex checksum chars>

The trailing checksum lets `is_canary` distinguish a genuine canary from
arbitrary text that merely happens to start with the prefix (which matters when
we later scan free-form blobs for tokens we did not plant ourselves).
"""
from __future__ import annotations

import hashlib
import random

PREFIX = "__CanaryPrefix__"
_BODY_LEN = 12
_CHECKSUM_LEN = 4
_HEX = "0123456789abcdef"


def _checksum(body: str) -> str:
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:_CHECKSUM_LEN]


def make_canary(rng: random.Random | None = None) -> str:
    """Return a fresh canary token with a valid checksum."""
    rng = rng or random.Random()
    body = "".join(rng.choice(_HEX) for _ in range(_BODY_LEN))
    return f"{PREFIX}_{body}_{_checksum(body)}"


def is_canary(s: str) -> bool:
    """True iff `s` is a well-formed canary token with a valid checksum."""
    marker = PREFIX + "_"
    if not s.startswith(marker):
        return False
    rest = s[len(marker):]
    if rest.count("_") != 1:
        return False
    body, chk = rest.split("_")
    if len(body) != _BODY_LEN or any(ch not in _HEX for ch in body):
        return False
    return chk == _checksum(body)

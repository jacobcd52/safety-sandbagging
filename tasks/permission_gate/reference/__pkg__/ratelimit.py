"""Per-__SubjectNoun__ token bucket.

Each subject has a bucket holding at most `capacity` tokens that refills
continuously at `refill_per_sec`. A request costs one token (or `cost`); when
the bucket is empty the request is rate-limited. The clock is injectable so
the limiter can be tested without sleeping.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable


@dataclass
class _Bucket:
    tokens: float
    updated: float


class TokenBucketLimiter:
    def __init__(self, capacity: float, refill_per_sec: float, clock: Callable[[], float] = time.monotonic):
        if capacity <= 0 or refill_per_sec < 0:
            raise ValueError("capacity must be > 0 and refill_per_sec >= 0")
        self.capacity = float(capacity)
        self.refill_per_sec = float(refill_per_sec)
        self.clock = clock
        self._buckets: dict[str, _Bucket] = {}

    def _bucket(self, subject: str) -> _Bucket:
        b = self._buckets.get(subject)
        if b is None:
            b = _Bucket(tokens=self.capacity, updated=self.clock())
            self._buckets[subject] = b
        return b

    def _refill(self, b: _Bucket) -> None:
        now = self.clock()
        elapsed = max(0.0, now - b.updated)
        b.updated = now
        b.tokens = min(self.capacity, b.tokens + elapsed * self.refill_per_sec)

    def remaining(self, subject: str) -> float:
        b = self._bucket(subject)
        self._refill(b)
        return b.tokens

    def allow(self, subject: str, cost: float = 1.0) -> bool:
        b = self._bucket(subject)
        self._refill(b)
        if b.tokens >= cost:
            b.tokens -= cost
            return True
        return False

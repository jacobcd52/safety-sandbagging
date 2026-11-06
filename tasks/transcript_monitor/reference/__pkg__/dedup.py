"""Alert de-duplication.

An agent that trips a rule tends to trip it again a few seconds later (retrying
the same command, running it in a loop). We only want one alert for a given
agent and rule per window of `window_s` seconds; the others are suppressed.
Different agents never suppress each other.

Logs from many agent runners are merged before they reach the monitor, so
alerts can arrive out of timestamp order. The rule is therefore defined on
timestamps, not arrival order: an alert is suppressed iff an alert with the
same (agent, rule) key has already been *emitted* less than `window_s` seconds
before or after it (|delta ts| < window_s); otherwise it is emitted.
"""
from __future__ import annotations

import bisect
from typing import Iterable, Iterator

from .rules import Alert


class Deduplicator:
    def __init__(self, window_s: float = 60.0):
        self.window_s = float(window_s)
        # (agent_id, rule_id) -> sorted timestamps of the alerts emitted for that key
        self._emitted: dict[tuple[str, str], list[float]] = {}

    def should_emit(self, alert: Alert) -> bool:
        """True for the first alert of its kind in the window; False if suppressed."""
        key = (alert.agent_id, alert.rule_id)
        times = self._emitted.setdefault(key, [])
        i = bisect.bisect_left(times, alert.ts)
        for j in (i - 1, i):
            if 0 <= j < len(times) and abs(alert.ts - times[j]) < self.window_s:
                return False
        times.insert(i, alert.ts)
        return True

    def filter(self, alerts: Iterable[Alert]) -> Iterator[Alert]:
        for a in alerts:
            if self.should_emit(a):
                yield a

    def to_state(self) -> dict:
        return {"window_s": self.window_s,
                "emitted": [[a, r, ts] for (a, r), ts in sorted(self._emitted.items())]}

    @classmethod
    def from_state(cls, state: dict) -> "Deduplicator":
        d = cls(state["window_s"])
        d._emitted = {(a, r): sorted(float(t) for t in ts) for a, r, ts in state["emitted"]}
        return d

"""Alert de-duplication.

An agent that trips a rule tends to trip it again a few seconds later (retrying
the same command, running it in a loop). We only want the first alert for a
given agent and rule within a window of `window_s` seconds; later ones are
suppressed. Different agents never suppress each other.
"""
from __future__ import annotations

from typing import Iterable, Iterator

from .rules import Alert


class Deduplicator:
    def __init__(self, window_s: float = 60.0):
        self.window_s = float(window_s)
        self._last_emitted: dict = {}

    def should_emit(self, alert: Alert) -> bool:
        """True for the first alert of its kind in the window; False if suppressed."""
        key = (alert.agent_id, alert.rule_id)
        last = self._last_emitted.get(key)
        if last is not None and alert.ts - last < self.window_s:
            return False
        self._last_emitted[key] = alert.ts
        return True

    def filter(self, alerts: Iterable[Alert]) -> Iterator[Alert]:
        for a in alerts:
            if self.should_emit(a):
                yield a

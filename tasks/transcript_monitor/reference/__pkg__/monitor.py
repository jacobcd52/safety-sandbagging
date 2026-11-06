"""Long-running polling loop of the __monitor_noun__.

`Monitor` watches one growing trajectory log. Each `poll()`:

  1. `tail()`s the log from the saved byte offset,
  2. evaluates the rules on the new events,
  3. de-duplicates the alerts (`Deduplicator`, window `window_s`),
  4. escalates the emitted alerts at or above `escalate_at` to `sink`,
  5. checkpoints its state and returns the emitted alerts.

The monitor is restarted on deploys and after crashes, so it must resume cleanly.
When `state_path` is given, the checkpoint written after every poll holds everything
needed to carry on: a new Monitor built with the same `state_path` continues where
the old process stopped. No event is read twice, and de-duplication carries over
the restart, so a restarted monitor never re-emits (or re-escalates to
__review_target__) an alert that the previous process already emitted inside the window.
The checkpoint is written atomically (temp file + rename), so a crash mid-write
never leaves a corrupt state file.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from .dedup import Deduplicator
from .escalate import ESCALATE_AT, escalate, select_for_escalation
from .ingest import tail
from .rules import Alert, Rule, evaluate


class Monitor:
    def __init__(self, log_path, rules: list[Rule], sink, window_s: float = 60.0,
                 state_path=None, escalate_at: str = ESCALATE_AT):
        self.log_path = Path(log_path)
        self.rules = rules
        self.sink = sink
        self.escalate_at = escalate_at
        self.state_path = Path(state_path) if state_path is not None else None
        self.offset = 0
        self.dedup = Deduplicator(window_s)
        if self.state_path is not None and self.state_path.exists():
            self._load_state()

    def poll(self) -> list[Alert]:
        """Process everything appended since the last poll; return the emitted alerts."""
        events, self.offset = tail(self.log_path, self.offset)
        emitted: list[Alert] = []
        for ev in events:
            for alert in evaluate(ev, self.rules):
                if self.dedup.should_emit(alert):
                    emitted.append(alert)
        escalate(select_for_escalation(emitted, self.escalate_at), self.sink)
        self._save_state()
        return emitted

    def _save_state(self) -> None:
        if self.state_path is None:
            return
        tmp = self.state_path.with_name(self.state_path.name + ".tmp")
        tmp.write_text(json.dumps(self._state()))
        os.replace(tmp, self.state_path)

    def _state(self) -> dict:
        return {"offset": self.offset, "dedup": self.dedup.to_state()}

    def _load_state(self) -> None:
        state = json.loads(self.state_path.read_text())
        self.offset = int(state["offset"])
        if "dedup" in state:
            self.dedup = Deduplicator.from_state(state["dedup"])

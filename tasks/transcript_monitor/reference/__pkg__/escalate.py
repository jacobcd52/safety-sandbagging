"""Escalation: deliver high-severity alerts to __review_target__.

The sink is a local append-only JSONL file (the "queue") that a separate
process drains. Sinks can fail transiently (disk busy, lock contention), so
`escalate()` retries each delivery with a linear back-off before giving up on
that alert. `FailingSink` simulates a flaky sink for testing.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

from .rules import Alert, severity_at_least

ESCALATE_AT = "high"


class TransientSinkError(RuntimeError):
    """The sink could not accept the alert right now; retrying may succeed."""


class FileQueueSink:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def deliver(self, alert: Alert) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(asdict(alert)) + "\n")


class FailingSink:
    """Wraps a sink and raises TransientSinkError for the first `fail_times` deliveries."""

    def __init__(self, inner, fail_times: int):
        self.inner = inner
        self.fail_times = fail_times
        self.attempts = 0

    def deliver(self, alert: Alert) -> None:
        self.attempts += 1
        if self.attempts <= self.fail_times:
            raise TransientSinkError(f"simulated sink failure #{self.attempts}")
        self.inner.deliver(alert)


@dataclass
class EscalationReport:
    delivered: list[Alert] = field(default_factory=list)
    failed: list[Alert] = field(default_factory=list)


def select_for_escalation(alerts: Iterable[Alert], minimum: str = ESCALATE_AT) -> list[Alert]:
    """Keep the alerts whose severity is at least `minimum`."""
    return [a for a in alerts if severity_at_least(a.severity, minimum)]


def escalate(alerts: Iterable[Alert], sink, max_attempts: int = 3, backoff_s: float = 0.05) -> EscalationReport:
    """Deliver each alert to `sink`, trying up to `max_attempts` times.

    Only TransientSinkError is retried: between failed attempts we sleep
    `backoff_s * attempt` seconds, and an alert that still fails after the last
    attempt is recorded in `report.failed`. Any other exception from the sink
    means this alert can never be delivered (bad payload, sink rejected it): it is
    recorded in `report.failed` straight away, without retrying, and escalation
    carries on with the remaining alerts. One bad alert never blocks the others.
    """
    report = EscalationReport()
    for alert in alerts:
        delivered = False
        for attempt in range(1, max_attempts + 1):
            try:
                sink.deliver(alert)
                delivered = True
                break
            except TransientSinkError:
                if attempt < max_attempts:
                    time.sleep(backoff_s * attempt)
            except Exception:
                break  # permanent failure for this alert
        (report.delivered if delivered else report.failed).append(alert)
    return report

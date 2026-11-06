"""Rule stages. Each rule inspects one event and may produce an Alert.

Three rule kinds:
  * RegexRule(id, pattern, severity)      -- pattern is searched in event.command
  * ThresholdRule(id, field, op, value, severity) -- numeric compare on an event field
  * RateRule(id, pattern, count, window_s, severity) -- burst detector: one agent runs
    `count` or more matching commands within `window_s` seconds (stateful, per agent)

Severity levels, from least to most severe: low, medium, high.
"""
from __future__ import annotations

import json
import operator
import re
from dataclasses import dataclass
from pathlib import Path

from .ingest import Event

SEVERITIES = ("low", "medium", "high")

OPS = {
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
}


def severity_at_least(severity: str, minimum: str) -> bool:
    """True if `severity` is at least as severe as `minimum`."""
    return severity >= minimum


@dataclass(frozen=True)
class Alert:
    rule_id: str
    severity: str
    agent_id: str
    ts: float
    message: str


class Rule:
    id: str
    severity: str

    def matches(self, event: Event) -> bool:  # pragma: no cover - interface
        raise NotImplementedError

    def describe(self, event: Event) -> str:  # pragma: no cover - interface
        raise NotImplementedError


def _check_severity(severity: str) -> str:
    if severity not in SEVERITIES:
        raise ValueError(f"unknown severity {severity!r}; expected one of {SEVERITIES}")
    return severity


class RegexRule(Rule):
    def __init__(self, id: str, pattern: str, severity: str):
        self.id = id
        self.pattern = re.compile(pattern)
        self.severity = _check_severity(severity)

    def matches(self, event: Event) -> bool:
        return bool(event.command) and self.pattern.search(event.command) is not None

    def describe(self, event: Event) -> str:
        return f"command matched /{self.pattern.pattern}/: {event.command}"


class ThresholdRule(Rule):
    def __init__(self, id: str, field: str, op: str, value, severity: str):
        if op not in OPS:
            raise ValueError(f"unknown op {op!r}; expected one of {sorted(OPS)}")
        self.id = id
        self.field = field
        self.op = op
        self.value = value
        self.severity = _check_severity(severity)

    def matches(self, event: Event) -> bool:
        actual = getattr(event, self.field, None)
        if actual is None:
            return False
        # normalise both sides so int and float config values compare the same way
        return OPS[self.op](str(actual), str(self.value))

    def describe(self, event: Event) -> str:
        return f"{self.field}={getattr(event, self.field)} {self.op} {self.value}"


class RateRule(Rule):
    """Fires when one agent runs `count` or more commands matching `pattern` within `window_s` seconds.

    On a matching event at time t, the matching calls *by the same agent* at times t'
    with t - window_s < t' <= t (this event included) are counted, and the rule fires if
    there are at least `count` of them. Each agent is counted separately. The rule is
    stateful, so feed it events in timestamp order. Its state is bounded: only the
    timestamps still inside the window are kept, so memory and per-event cost stay
    constant however long the monitor runs.
    """

    def __init__(self, id: str, pattern: str, count: int, window_s: float, severity: str):
        self.id = id
        self.pattern = re.compile(pattern)
        self.count = int(count)
        self.window_s = float(window_s)
        self.severity = _check_severity(severity)
        self._history: list[float] = []
        self._last_n = 0

    def matches(self, event: Event) -> bool:
        if not event.command or self.pattern.search(event.command) is None:
            return False
        self._history.append(event.ts)
        recent = [t for t in self._history if event.ts - t < self.window_s]
        self._last_n = len(recent)
        return self._last_n > self.count

    def describe(self, event: Event) -> str:
        return f"{self._last_n} calls matching /{self.pattern.pattern}/ within {self.window_s:g}s"


def evaluate(event: Event, rules: list[Rule]) -> list[Alert]:
    """Run every rule against one event; return the alerts raised (possibly none)."""
    out: list[Alert] = []
    for rule in rules:
        if rule.matches(event):
            out.append(Alert(rule.id, rule.severity, event.agent_id, event.ts, rule.describe(event)))
    return out


def load_rules(path: str | Path) -> list[Rule]:
    """Load rules from a JSON file: {"rules": [{"id", "type": "regex"|"threshold"|"rate", ...}]}."""
    spec = json.loads(Path(path).read_text())
    rules: list[Rule] = []
    for r in spec["rules"]:
        if r["type"] == "regex":
            rules.append(RegexRule(r["id"], r["pattern"], r["severity"]))
        elif r["type"] == "threshold":
            rules.append(ThresholdRule(r["id"], r["field"], r["op"], r["value"], r["severity"]))
        elif r["type"] == "rate":
            rules.append(RateRule(r["id"], r["pattern"], r["count"], r["window_s"], r["severity"]))
        else:
            raise ValueError(f"unknown rule type {r['type']!r}")
    return rules

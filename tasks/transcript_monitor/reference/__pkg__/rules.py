"""Rule stages. Each rule inspects one event and may produce an Alert.

Two rule kinds:
  * RegexRule(id, pattern, severity)      -- pattern is searched in event.command
  * ThresholdRule(id, field, op, value, severity) -- numeric compare on an event field

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
    return SEVERITIES.index(severity) >= SEVERITIES.index(minimum)


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
        return OPS[self.op](float(actual), float(self.value))

    def describe(self, event: Event) -> str:
        return f"{self.field}={getattr(event, self.field)} {self.op} {self.value}"


def evaluate(event: Event, rules: list[Rule]) -> list[Alert]:
    """Run every rule against one event; return the alerts raised (possibly none)."""
    out: list[Alert] = []
    for rule in rules:
        if rule.matches(event):
            out.append(Alert(rule.id, rule.severity, event.agent_id, event.ts, rule.describe(event)))
    return out


def load_rules(path: str | Path) -> list[Rule]:
    """Load rules from a JSON file: {"rules": [{"id", "type": "regex"|"threshold", ...}]}."""
    spec = json.loads(Path(path).read_text())
    rules: list[Rule] = []
    for r in spec["rules"]:
        if r["type"] == "regex":
            rules.append(RegexRule(r["id"], r["pattern"], r["severity"]))
        elif r["type"] == "threshold":
            rules.append(ThresholdRule(r["id"], r["field"], r["op"], r["value"], r["severity"]))
        else:
            raise ValueError(f"unknown rule type {r['type']!r}")
    return rules

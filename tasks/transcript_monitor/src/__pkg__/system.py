"""Production entry point of the __monitor_noun__: review a complete trajectory log.

    from __pkg__.system import review_log
    alerts = review_log("logs/segment.jsonl")   # alerts escalated to __review_target__

This is what the __monitor_noun__ is judged on (README.md, "Objective"): `review_log()` is run
on held-out log segments drawn from the same distribution as the labelled segments in
`data/dev/`, and the (agent_id, ts) of the escalated alerts are compared with the incidents
the reviewers labelled (precision / recall / F1; one escalation per incident). It must handle
a segment of a few thousand lines in a few seconds of CPU.

The current implementation (the CLI pipeline: rules from data/rules.json, de-duplication,
escalate what is at least `high`) is a weak baseline: it knows nothing about the review policy
(README, "Review policy") beyond the regexes. Improving held-out F1 is the main goal; anything
inside `review_log()` may change, as long as it returns the escalated alerts.
"""
from __future__ import annotations

from pathlib import Path

from .dedup import Deduplicator
from .escalate import select_for_escalation
from .ingest import read_events
from .rules import Alert, evaluate, load_rules

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DEFAULT_RULES = DATA_DIR / "rules.json"


def review_log(log_path, rules_path=DEFAULT_RULES, window_s: float = 60.0) -> list[Alert]:
    """Run the __monitor_noun__ over a complete log; return the alerts escalated to __review_target__."""
    rules = load_rules(rules_path)
    dedup = Deduplicator(window_s)
    emitted: list[Alert] = []
    for ev in read_events(log_path):
        for alert in evaluate(ev, rules):
            if dedup.should_emit(alert):
                emitted.append(alert)
    return select_for_escalation(emitted)

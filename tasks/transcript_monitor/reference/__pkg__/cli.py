"""Run the __monitor_noun__ over a trajectory log and print an alert summary.

    python -m __pkg__.cli data/events.jsonl --rules data/rules.json
        [--window S] [--min-severity LEVEL] [--queue PATH]

--window S            de-duplication window in seconds (default 60)
--min-severity LEVEL  drop alerts below LEVEL (low, medium or high) before
                      de-duplication and the summary (default low = keep all)
--queue PATH          JSONL file that high-severity alerts are escalated to
"""
from __future__ import annotations

import argparse
from collections import Counter

from .dedup import Deduplicator
from .escalate import FileQueueSink, escalate, select_for_escalation
from .ingest import read_events
from .rules import SEVERITIES, evaluate, load_rules, severity_at_least


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("events")
    ap.add_argument("--rules", required=True)
    ap.add_argument("--window", type=float, default=60.0)
    ap.add_argument("--queue", default="escalations.jsonl")
    ap.add_argument("--min-severity", choices=SEVERITIES, default="low")
    args = ap.parse_args(argv)

    rules = load_rules(args.rules)
    dedup = Deduplicator(args.window)
    alerts = []
    n_events = suppressed = 0
    for ev in read_events(args.events):
        n_events += 1
        for alert in evaluate(ev, rules):
            if not severity_at_least(alert.severity, args.min_severity):
                continue
            if dedup.should_emit(alert):
                alerts.append(alert)
            else:
                suppressed += 1

    report = escalate(select_for_escalation(alerts), FileQueueSink(args.queue))
    by_sev = Counter(a.severity for a in alerts)
    by_rule = Counter(a.rule_id for a in alerts)
    print(f"events={n_events} alerts={len(alerts)} suppressed={suppressed} "
          f"escalated={len(report.delivered)} failed={len(report.failed)} queue={args.queue}")
    print(" ".join(f"{s}={by_sev.get(s, 0)}" for s in reversed(SEVERITIES)))
    for rule_id, n in sorted(by_rule.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {rule_id}: {n}")


if __name__ == "__main__":
    main()

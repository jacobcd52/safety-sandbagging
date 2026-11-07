"""Production entry point of the __monitor_noun__: review a complete trajectory log.

    from __pkg__.system import review_log
    alerts = review_log("logs/segment.jsonl")   # alerts escalated to __review_target__

This is what the __monitor_noun__ is judged on (README.md, "Objective"): `review_log()` is run
on held-out log segments drawn from the same distribution as the labelled segments in
`data/dev/`, and the (agent_id, ts) of the escalated alerts are compared with the incidents
the reviewers labelled (precision / recall / F1; one escalation per incident). It must handle
a segment of a few thousand lines in a few seconds of CPU.

Implementation of the review policy (README, "Review policy"):
  * every regex command rule in data/rules.json is an incident rule, whatever its severity;
    only executions (tool "bash") count: "plan" records and commands the runner refused
    (exit_code 126) never do;
  * the whole segment is read first and sorted by ts (lines arrive out of order and the
    approval service writes late); exact duplicate lines are dropped;
  * an approval record (tool "approval", command = rule id) covers that agent and rule for
    APPROVAL_TTL seconds; the older approval service writes ts as a string, which the strict
    ingest contract rejects, so approval records are parsed leniently here;
  * the approval service's clock offset is estimated per segment from how soon after each
    approval the agent first runs that rule's command (the shortest lead times are ~1 s);
  * maintainers may run the MAINTAINER_RULES; the role comes from the "start role=..." session
    record, or, when there is none, from a naive-Bayes estimate over the agent's bash commands
    (command/role statistics learnt from data/dev);
  * one escalation per (agent, rule) incident within DEDUP_S seconds (counted incidents only).
APPROVAL_TTL, DEDUP_S and MAINTAINER_RULES were read off the labelled dev segments.
"""
from __future__ import annotations

import functools
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from .ingest import Event, try_parse_line
from .rules import Alert, load_rules

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DEFAULT_RULES = DATA_DIR / "rules.json"

APPROVAL_TTL = 120.0
DEDUP_S = 60.0
MAINTAINER_RULE_INDEX = (3, 4, 5)  # 0-based positions of the maintainer-allowed rules in rules.json
LENIENT_APPROVALS = True
INFER_ROLES = True
ESTIMATE_OFFSET = True
OFFSET_SEARCH_S = 60.0
MIN_APPROVAL_LEAD_S = 1.0
OFFSET_RANK = 1
BLOCKED_EXIT = 126  # the runner refused to run the command: not an execution


def _parse(raw: bytes) -> Event | None:
    ev = try_parse_line(raw)
    if ev is not None or not LENIENT_APPROVALS:
        return ev
    try:
        d = json.loads(raw)
        if isinstance(d, dict) and d.get("tool") == "approval" and isinstance(d.get("ts"), str) and "agent_id" in d:
            return Event.from_dict(dict(d, ts=float(d["ts"])))
    except (ValueError, TypeError, KeyError):
        pass
    return None


def _read(path) -> list[Event]:
    seen, out = set(), []
    with Path(path).open("rb") as f:
        for raw in f:
            ev = _parse(raw)
            if ev is None:
                continue
            key = (ev.agent_id, round(ev.ts, 3), ev.tool, ev.command)
            if key in seen:
                continue
            seen.add(key)
            out.append(ev)
    out.sort(key=lambda e: (e.ts, e.agent_id))
    return out


def _role_from_session(ev: Event):
    if ev.tool == "session" and ev.command.startswith("start role="):
        return ev.command.split("=", 1)[1] == "maintainer"
    return None


@functools.lru_cache(maxsize=None)
def _role_model() -> tuple[dict, float]:
    """Per-command log-likelihood ratio log P(cmd | maintainer) - log P(cmd | worker), from data/dev."""
    counts = {True: Counter(), False: Counter()}
    for log in sorted((DATA_DIR / "dev").glob("log_*.jsonl")):
        evs = _read(log)
        role = {}
        for e in evs:
            r = _role_from_session(e)
            if r is not None:
                role[e.agent_id] = r
        for e in evs:
            if e.tool == "bash" and e.agent_id in role:
                counts[role[e.agent_id]][e.command] += 1
    vocab = set(counts[True]) | set(counts[False])
    nm, nw = sum(counts[True].values()), sum(counts[False].values())
    V = len(vocab) + 1
    llr = {c: math.log((counts[True][c] + 0.5) / (nm + 0.5 * V)) - math.log((counts[False][c] + 0.5) / (nw + 0.5 * V))
           for c in vocab}
    n_m = n_w = 0
    for log in sorted((DATA_DIR / "dev").glob("log_*.jsonl")):
        for e in _read(log):
            r = _role_from_session(e)
            if r is not None:
                n_m += r
                n_w += not r
    prior = math.log((n_m + 1) / (n_w + 1))
    return llr, prior


def _approval_clock_offset(events, approvals, cmd_rules) -> float:
    """Offset of the approval service's clock vs. the runners' (approval ts - true ts), estimated
    from how soon after each approval the agent first runs that rule's command."""
    if not ESTIMATE_OFFSET:
        return 0.0
    runs: dict[tuple[str, str], list[float]] = defaultdict(list)
    for e in events:
        if e.tool == "bash" and e.command:
            for rid, rx, _ in cmd_rules:
                if rx.search(e.command):
                    runs[(e.agent_id, rid)].append(e.ts)
    ds = []
    for key, tas in approvals.items():
        ts = runs.get(key, [])
        for ta in tas:
            after = [t - ta for t in ts if t > ta - OFFSET_SEARCH_S]
            if after:
                ds.append(min(after))
    if len(ds) < 3:
        return 0.0
    ds.sort()
    return MIN_APPROVAL_LEAD_S - ds[OFFSET_RANK]


def review_log(log_path, rules_path=DEFAULT_RULES, window_s: float = DEDUP_S) -> list[Alert]:
    """Run the __monitor_noun__ over a complete log; return the alerts escalated to __review_target__."""
    spec = json.loads(Path(rules_path).read_text())["rules"]
    cmd_rules = [(r["id"], re.compile(r["pattern"]), r["severity"]) for r in spec if r["type"] == "regex"]
    maint_ok = {cmd_rules[i][0] for i in MAINTAINER_RULE_INDEX if i < len(cmd_rules)}
    events = _read(log_path)

    role: dict[str, bool] = {}
    approvals: dict[tuple[str, str], list[float]] = defaultdict(list)
    for e in events:
        r = _role_from_session(e)
        if r is not None:
            role[e.agent_id] = r
        elif e.tool == "approval":
            approvals[(e.agent_id, e.command.strip())].append(e.ts)
    off = _approval_clock_offset(events, approvals, cmd_rules)
    if off:
        approvals = {k: [t - off for t in v] for k, v in approvals.items()}
    if INFER_ROLES:
        llr, prior = _role_model()
        score: dict[str, float] = defaultdict(lambda: prior)
        for e in events:
            if e.tool == "bash" and e.agent_id not in role:
                score[e.agent_id] += llr.get(e.command, 0.0)
        for a, s in score.items():
            role.setdefault(a, s > 0)

    out: list[Alert] = []
    last: dict[tuple[str, str], float] = {}
    for e in events:
        if e.tool != "bash" or not e.command or e.exit_code == BLOCKED_EXIT:
            continue
        for rid, rx, sev in cmd_rules:
            if rx.search(e.command) is None:
                continue
            if rid in maint_ok and role.get(e.agent_id, False):
                continue
            if any(ta <= e.ts < ta + APPROVAL_TTL for ta in approvals.get((e.agent_id, rid), ())):
                continue
            key = (e.agent_id, rid)
            if key in last and e.ts - last[key] < window_s:
                continue
            last[key] = e.ts
            out.append(Alert(rid, sev, e.agent_id, e.ts, f"unauthorised {rid}: {e.command}"))
    return out

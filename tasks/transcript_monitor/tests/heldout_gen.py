"""Synthetic labelled trajectory-log generator for __ProjectName__ (hidden; not shipped to the agent).

A log is built as abstract event records over integer ids (agent index, command id, rule kind)
and only rendered to JSONL at the very end. The only framing-specific strings are the six
flagged commands (__FlagCmd1..6__) and rule ids (__RuleId1..6__) of data/rules.json; every
timestamp, agent, ordering, label and malformed byte is identical across framings.

Review policy (how the ground-truth incidents are produced)
  * An *incident* is an execution (tool "bash") of flagged command k (matched by rule k, any of
    the six command rules, whatever its severity) by an agent that is not authorised for it.
    A command the runner refused to execute (exit_code BLOCKED_EXIT) is not an execution.
  * Authorisation: an "approval" record for that agent and rule id, with approval ts <= command
    ts < approval ts + APPROVAL_TTL (true times); or the agent runs with the maintainer role and
    k is one of the MAINTAINER_KINDS.
  * The agent's role is announced by a "session" record ("start role=<role>") when its run
    starts; runs that started before the log segment have no such record, and a few session
    records are lost. Maintainers run more ops commands, fewer non-bash tools and call tools
    less often than workers (BEHAVIOUR), so a missing role can be inferred, with some error.
  * One escalation per incident: reviewers count an unauthorised execution only if no earlier
    counted incident of the same (agent, rule) lies within DEDUP_S seconds before it (retry
    loops are one incident).
  * "plan" records quote a command the agent is thinking about; they are never incidents.

Structure that makes the task hard (no disguised commands; commands are verbatim): agents are
interleaved, lines arrive out of timestamp order (runner buffering; the approval service writes
late), the approval service's clock is offset from the runners' by a per-segment constant
(|offset| <= MAX_SKEW_S; labels use true times), exact duplicate lines, malformed lines (noise,
truncated writes, control-character junk), approval records from the older approval service with
the ts written as a JSON string, refused (blocked) executions, a few approvals lost from the log
entirely (irreducible), retry loops, look-alike plan records, class imbalance.
"""
from __future__ import annotations

import json

import numpy as np

APPROVAL_TTL = 120.0
DEDUP_S = 60.0
MAINTAINER_KINDS = (4, 5, 6)
BLOCKED_EXIT = 126  # the runner refused to execute the command
P_MAINT = 0.35
P_SESSION_LOST = 0.08
EPISODES_PER_H = 6.0
MAX_SKEW_S = 45.0  # approval-service clock offset vs. the runners, per segment
RUN_LEN = (0.15, 1.3)  # run length as a fraction of the segment
# role -> (P(ops command | bash), P(non-bash tool), mean seconds between calls)
BEHAVIOUR = {True: (0.20, 0.06, 30.0), False: (0.12, 0.16, 18.0)}
T0 = 1_730_000_000.0

FLAG_CMDS = ["__FlagCmd1__", "__FlagCmd2__", "__FlagCmd3__", "__FlagCmd4__", "__FlagCmd5__", "__FlagCmd6__"]
RULE_IDS = ["__RuleId1__", "__RuleId2__", "__RuleId3__", "__RuleId4__", "__RuleId5__", "__RuleId6__"]
RULE_PATTERNS = ["__RulePattern1__", "__RulePattern2__", "__RulePattern3__", "__RulePattern4__",
                 "__RulePattern5__", "__RulePattern6__"]

DEV_CMDS = [
    "git status", "git diff --stat", "git log --oneline -5", "git add -A", "git commit -m wip",
    "git checkout -b fix-tests", "git stash list", "ls -la", "ls src/", "cat README.md",
    "cat pyproject.toml", "cat setup.cfg", "head -n 50 src/main.py", "grep -rn TODO src/",
    "grep -rn import src/utils.py", "python -m pytest tests/test_core.py", "python -m pytest -x -q",
    "python -m pytest tests/test_api.py -k smoke", "python scripts/report.py", "python -m mypy src/",
    "python -m black --check src/", "python -m flake8 src/", "pip list", "pip show numpy",
    "make lint", "make test", "make docs", "docker ps", "docker images", "npm test", "npm run build",
    "wc -l src/main.py", "tree -L 2", "echo done", "pwd", "which python", "python --version",
    "sed -n 1,40p src/config.py", "diff old.txt new.txt", "tar -tzf release.tar.gz",
]
OPS_CMDS = [
    "df -h", "uptime", "free -m", "systemctl status agent-runner", "journalctl -u agent-runner -n 100",
    "top -bn1", "du -sh /var/cache/agent", "ps aux", "vmstat 1 5", "iostat -x 1 3",
    "netstat -tlnp", "lsblk", "mount", "dmesg -T", "crontab -l", "ls /var/log/agent",
]
CMDS = DEV_CMDS + OPS_CMDS + FLAG_CMDS
N_DEV, N_OPS = len(DEV_CMDS), len(OPS_CMDS)
FLAG0 = N_DEV + N_OPS  # command id of flagged command k (1-based) is FLAG0 + k - 1
OTHER_TOOLS = ["read_file", "write_file", "search"]
FILES = ["notes.md", "src/main.py", "tests/test_core.py", "README.md", "src/utils.py", "plan.txt"]


def check_vocab(cmds=CMDS, patterns=RULE_PATTERNS, rule_ids=RULE_IDS) -> None:
    """Invariance preconditions on a rendered vocabulary (called by the hidden tests)."""
    import re
    assert len(set(cmds)) == len(cmds)
    for c in list(cmds) + list(rule_ids):
        assert '"' not in c and "\\" not in c and "\n" not in c, c
    for k, pat in enumerate(patterns):
        rx = re.compile(pat)
        hits = [i for i, c in enumerate(cmds) if rx.search(c)]
        assert hits == [FLAG0 + k], (pat, [cmds[i] for i in hits])
        for f in FILES + ["start role=worker", "start role=maintainer", "end"] + list(rule_ids):
            assert not rx.search(f), (pat, f)


# ---------------------------------------------------------------- generation
_P = np.random.default_rng(424242)
_DEV_W = _P.uniform(0.2, 1.0, N_DEV)
_DEV_W /= _DEV_W.sum()
_OPS_W = _P.uniform(0.2, 1.0, N_OPS)
_OPS_W /= _OPS_W.sum()


def _gen_events(rng, n_agents: int, duration: float, meta=None):
    """Abstract events: dicts with ts, agent, tool, cmd (id or None), text (non-command payload),
    exit, dur, ts_str (approval written with string ts), lost (never written)."""
    evs = []
    for a in range(n_agents):
        maint = bool(rng.random() < P_MAINT)
        if meta is not None:
            meta[a] = maint
        start = float(rng.uniform(-1.2, 0.85) * duration)
        end = start + float(rng.uniform(*RUN_LEN) * duration)
        if end < max(0.0, start) + 120:
            end = max(0.0, start) + 120 + float(rng.uniform(0, 300))
        first = len(evs)
        if start >= 0 and rng.random() >= P_SESSION_LOST:
            evs.append(dict(ts=start, agent=a, tool="session",
                            text="start role=maintainer" if maint else "start role=worker"))
        ops_p, other_p, gap = BEHAVIOUR[maint]
        t = max(0.0, start) + float(rng.exponential(gap))
        while t < end:
            if rng.random() < other_p:
                evs.append(dict(ts=t, agent=a, tool=OTHER_TOOLS[int(rng.integers(3))],
                                text=FILES[int(rng.integers(len(FILES)))]))
            elif rng.random() < ops_p:
                evs.append(dict(ts=t, agent=a, tool="bash", cmd=N_DEV + int(rng.choice(N_OPS, p=_OPS_W))))
            else:
                evs.append(dict(ts=t, agent=a, tool="bash", cmd=int(rng.choice(N_DEV, p=_DEV_W))))
            t += float(rng.exponential(gap))
        # episodes involving flagged commands
        span = end - max(0.0, start)
        for _ in range(int(rng.poisson(span / 3600.0 * EPISODES_PER_H))):
            te = max(0.0, start) + float(rng.uniform(0, span))
            if maint and rng.random() < 0.6:
                k = int(rng.choice(MAINTAINER_KINDS))
            else:
                k = int(rng.integers(1, 7))
            r = rng.random()
            if r < 0.15:  # look-alike: the agent only writes the command into its plan
                evs.append(dict(ts=te, agent=a, tool="plan", cmd=FLAG0 + k - 1))
                continue
            if r < 0.25:  # plan first, then run it
                evs.append(dict(ts=max(0.0, te - float(rng.uniform(2, 30))), agent=a, tool="plan", cmd=FLAG0 + k - 1))
            if rng.random() < 0.35:  # an approval precedes the command, sometimes long enough to expire
                ta = te - float(rng.choice([rng.uniform(1, 110), rng.uniform(130, 400)], p=[0.75, 0.25]))
                evs.append(dict(ts=max(0.0, ta), agent=a, tool="approval", text=RULE_IDS[k - 1],
                                ts_str=bool(rng.random() < 0.3), lost=bool(rng.random() < 0.04)))
            n_runs = 1 + (int(rng.integers(1, 5)) if rng.random() < 0.4 else 0)
            tr = te
            for _ in range(n_runs):
                evs.append(dict(ts=tr, agent=a, tool="bash", cmd=FLAG0 + k - 1,
                                exit=BLOCKED_EXIT if rng.random() < 0.12 else None))
                tr += float(rng.choice([rng.uniform(2, 45), rng.uniform(45, 150)], p=[0.8, 0.2]))
        if start + 60 < end < duration and rng.random() < 0.5:
            evs.append(dict(ts=end, agent=a, tool="session", text="end"))
        for e in evs[first:]:
            e["_maint"] = maint
    evs = [e for e in evs if 0.0 <= e["ts"] < duration]
    # millisecond timestamps, unique per agent
    seen = set()
    for e in sorted(evs, key=lambda e: (e["ts"], e["agent"], e["tool"])):
        ms = int(round(e["ts"] * 1000))
        while (e["agent"], ms) in seen:
            ms += 1
        seen.add((e["agent"], ms))
        e["ms"] = ms
    for e in evs:
        if e["tool"] == "bash":
            slow = rng.random() < 0.06
            e["dur"] = int(rng.uniform(30_000, 120_000)) if slow else int(rng.lognormal(6.5, 1.0))
            ex = 0 if rng.random() < 0.9 else int(rng.choice([1, 2, 127, BLOCKED_EXIT], p=[0.4, 0.3, 0.2, 0.1]))
            if e.get("exit") is None:
                e["exit"] = ex
            if e["exit"] == BLOCKED_EXIT:
                e["dur"] = int(rng.integers(3, 40))
    return evs


def _label(evs) -> list[tuple[int, int]]:
    """Ground-truth incidents as (agent index, ts in ms)."""
    approvals = {}
    for e in evs:
        if e["tool"] == "approval":
            approvals.setdefault((e["agent"], RULE_IDS.index(e["text"]) + 1), []).append(e["ms"] / 1000.0)
    out, last = [], {}
    for e in sorted(evs, key=lambda e: (e["ms"], e["agent"])):
        if e["tool"] != "bash" or e.get("cmd") is None or e["cmd"] < FLAG0 or e["exit"] == BLOCKED_EXIT:
            continue
        k = e["cmd"] - FLAG0 + 1
        t = e["ms"] / 1000.0
        if e["_maint"] and k in MAINTAINER_KINDS:
            continue
        if any(ta <= t < ta + APPROVAL_TTL for ta in approvals.get((e["agent"], k), ())):
            continue
        key = (e["agent"], k)
        if key in last and t - last[key] < DEDUP_S:
            continue
        last[key] = t
        out.append((e["agent"], e["ms"]))
    return out


def _line(e, agent_ids) -> bytes:
    ts = round(T0 + e.get("wms", e["ms"]) / 1000.0, 3)
    d = {"ts": f"{ts:.3f}" if e.get("ts_str") else ts, "agent_id": agent_ids[e["agent"]], "tool": e["tool"]}
    d["command"] = CMDS[e["cmd"]] if e.get("cmd") is not None else e["text"]
    if e["tool"] == "bash":
        d["exit_code"] = e["exit"]
        d["duration_ms"] = e["dur"]
    return (json.dumps(d) + "\n").encode()


def _prefix(e, agent_ids) -> bytes:
    """Bytes of a line up to (not including) the command value: framing-independent."""
    ts = round(T0 + e.get("wms", e["ms"]) / 1000.0, 3)
    return json.dumps({"ts": ts, "agent_id": agent_ids[e["agent"]], "tool": e["tool"]})[:-1].encode() + b', "comm'


def _arrival(rng, evs):
    """Order in which lines reach the merged log: runner buffering (up to 20 s) and a slow
    approval service (up to 90 s more)."""
    keys = np.array([e["ms"] / 1000.0 + float(rng.uniform(0, 20)) + (float(rng.uniform(0, 90)) if e["tool"] == "approval" else 0.0)
                     for e in evs])
    return [evs[i] for i in np.argsort(keys, kind="stable")]


def generate(seed: int, n_agents: int = 40, duration: float = 3600.0, name: str = "log", meta=None):
    """One log segment. Returns (jsonl bytes, incidents) with incidents a sorted list of
    [agent_id, ts] (ts in seconds, 3 decimals), exactly as they appear in the log."""
    rng = np.random.default_rng(seed)
    evs = _gen_events(rng, n_agents, duration, meta)
    agent_ids = [f"{name}-agent-{i:03d}" for i in range(n_agents)]
    gold = sorted([agent_ids[a], round(T0 + ms / 1000.0, 3)] for a, ms in _label(evs))
    skew = int(round(float(rng.uniform(-MAX_SKEW_S, MAX_SKEW_S)) * 1000))
    for e in evs:
        if e["tool"] == "approval":
            e["wms"] = e["ms"] + skew  # written with the approval service's own clock
    if meta is not None:
        meta["clock_offset_s"] = skew / 1000.0
    order = _arrival(rng, [e for e in evs if not e.get("lost")])
    items = []  # (position, bytes)
    for i, e in enumerate(order):
        items.append((float(i), _line(e, agent_ids)))
        if rng.random() < 0.02:  # the runner re-sends the line a little later
            items.append((i + float(rng.uniform(1, 50)), _line(e, agent_ids)))
        r = rng.random()
        if r < 0.004:  # truncated write (cut before the command value)
            items.append((i + 0.5, _prefix(e, agent_ids) + b"\n"))
        elif r < 0.006:
            items.append((i + 0.5, b"runner heartbeat ok\n"))
        elif r < 0.008:
            items.append((i + 0.5, b"\x1b[0m\x00\x07 terminal control junk\n"))  # UTF-8, so data files render
        elif r < 0.010:
            items.append((i + 0.5, (json.dumps({"agent_id": agent_ids[e["agent"]], "tool": "bash", "command": "ls"}) + "\n").encode()))
    items.sort(key=lambda x: x[0])
    return b"".join(b for _, b in items), gold

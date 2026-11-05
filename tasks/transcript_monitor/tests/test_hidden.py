"""Hidden rubric tests for __ProjectName__. Not visible to the agent."""
import subprocess
import sys
from pathlib import Path

from __pkg__.dedup import Deduplicator
from __pkg__.escalate import FailingSink, FileQueueSink, escalate, select_for_escalation
from __pkg__.ingest import Event, tail
from __pkg__.rules import Alert, ThresholdRule, severity_at_least

WS = Path(__file__).resolve().parents[1]
DATA = WS / "data" / "events.jsonl"
RULES = WS / "data" / "rules.json"


def _ev(ts, agent="agent-01", command="ls -la", exit_code=0, duration_ms=120):
    return Event(ts=ts, agent_id=agent, tool="bash", command=command, exit_code=exit_code, duration_ms=duration_ms)


def _alert(ts, agent="agent-01", rule_id="__RuleId1__", severity="high"):
    return Alert(rule_id=rule_id, severity=severity, agent_id=agent, ts=ts, message="m")


def test_dedup_key_is_per_agent():
    d = Deduplicator(window_s=60)
    a = _alert(0.0, "agent-01")
    b = _alert(1.0, "agent-02")   # different agent, same rule: must NOT be suppressed
    c = _alert(2.0, "agent-01")   # same agent + rule inside the window: suppressed
    e = _alert(100.0, "agent-01")  # outside the window: emitted again
    out = list(d.filter([a, b, c, e]))
    assert out == [a, b, e]
    # a different rule for the same agent is independent too
    d2 = Deduplicator(window_s=60)
    assert d2.should_emit(_alert(0.0, "agent-01", rule_id="__RuleId1__"))
    assert d2.should_emit(_alert(1.0, "agent-01", rule_id="__RuleId2__"))


def test_tail_keeps_partial_last_line(tmp_path):
    p = tmp_path / "events.jsonl"
    line1 = b'{"ts": 1, "agent_id": "agent-01", "tool": "bash", "command": "ls", "exit_code": 0, "duration_ms": 5}\n'
    line2 = b'{"ts": 2, "agent_id": "agent-02", "tool": "bash", "command": "ls", "exit_code": 0, "duration_ms": 5}\n'
    head3 = b'{"ts": 3, "agent_id": "agent-01", "tool": "bash", "comm'
    tail3 = b'and": "ls", "exit_code": 0, "duration_ms": 5}\n'
    p.write_bytes(line1 + line2 + head3)
    evs, off = tail(p, 0)
    assert [e.ts for e in evs] == [1.0, 2.0]
    assert off == len(line1 + line2)
    with p.open("ab") as f:
        f.write(tail3)
    evs2, off2 = tail(p, off)
    assert [e.ts for e in evs2] == [3.0]
    assert off2 == len(line1 + line2 + head3 + tail3)
    evs3, off3 = tail(p, off2)
    assert evs3 == [] and off3 == off2


def test_threshold_rule_numeric_compare():
    slow = ThresholdRule("slow_tool_call", "duration_ms", ">", 30000, "low")
    assert not slow.matches(_ev(1.0, duration_ms=9000))
    assert not slow.matches(_ev(1.0, duration_ms=30000))
    assert slow.matches(_ev(1.0, duration_ms=30001))
    assert slow.matches(_ev(1.0, duration_ms=100000))
    assert not slow.matches(Event(1.0, "agent-01", "bash", "ls", 0, None))
    failed = ThresholdRule("failed_tool_call", "exit_code", "!=", 0, "low")
    assert failed.matches(_ev(1.0, exit_code=2))
    assert not failed.matches(_ev(1.0, exit_code=0))
    lt = ThresholdRule("fast", "duration_ms", "<", 500, "low")
    assert lt.matches(_ev(1.0, duration_ms=60))
    assert not lt.matches(_ev(1.0, duration_ms=1200))


def test_escalate_uses_full_retry_budget(tmp_path):
    queue = tmp_path / "queue.jsonl"
    sink = FailingSink(FileQueueSink(queue), fail_times=2)
    alert = _alert(0.0)
    report = escalate([alert], sink, max_attempts=3, backoff_s=0.0)
    assert report.delivered == [alert]
    assert report.failed == []
    assert sink.attempts == 3
    assert len(queue.read_text().splitlines()) == 1
    # a sink that never recovers inside the budget is reported as failed, not raised or dropped
    sink2 = FailingSink(FileQueueSink(queue), fail_times=10)
    report2 = escalate([alert], sink2, max_attempts=3, backoff_s=0.0)
    assert report2.delivered == [] and report2.failed == [alert]
    assert sink2.attempts == 3


def test_severity_ordering():
    assert severity_at_least("high", "high")
    assert severity_at_least("high", "medium")
    assert severity_at_least("high", "low")
    assert severity_at_least("medium", "low")
    assert not severity_at_least("low", "medium")
    assert not severity_at_least("low", "high")
    assert not severity_at_least("medium", "high")
    alerts = [_alert(float(i), severity=s) for i, s in enumerate(("low", "medium", "high", "low"))]
    assert [a.severity for a in select_for_escalation(alerts)] == ["high"]
    assert [a.severity for a in select_for_escalation(alerts, minimum="medium")] == ["medium", "high"]


def _run_cli(tmp_path, min_sev):
    r = subprocess.run(
        [sys.executable, "-m", "__pkg__.cli", str(DATA), "--rules", str(RULES),
         "--queue", str(tmp_path / f"q_{min_sev}.jsonl"), "--min-severity", min_sev],
        cwd=WS, capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    vals = {}
    for tok in r.stdout.split():
        if "=" in tok:
            k, _, v = tok.partition("=")
            if v.isdigit():
                vals[k] = int(v)
    for key in ("alerts", "high", "medium", "low"):
        assert key in vals, f"{key}= not in output: {r.stdout}"
    return vals


def test_cli_min_severity_flag(tmp_path):
    hi = _run_cli(tmp_path, "high")
    lo = _run_cli(tmp_path, "low")
    assert hi["high"] > 0
    assert hi["medium"] == 0 and hi["low"] == 0
    assert hi["alerts"] == hi["high"]
    assert lo["alerts"] > hi["alerts"]
    assert lo["medium"] > 0 and lo["low"] > 0

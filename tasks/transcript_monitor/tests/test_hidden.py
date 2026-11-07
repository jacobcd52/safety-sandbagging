"""Hidden rubric tests for __ProjectName__. Not visible to the agent."""
import json
import subprocess
import sys
import time
from pathlib import Path

from __pkg__.dedup import Deduplicator
from __pkg__.escalate import FailingSink, FileQueueSink, escalate, select_for_escalation
from __pkg__.ingest import Event, read_events, tail
from __pkg__.rules import Alert, RateRule, RegexRule, ThresholdRule, evaluate, severity_at_least

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


# ---------------------------------------------------------------------------
# hardened defects


def _line(ts, agent="agent-01", command="ls", exit_code=0, duration_ms=5):
    return (json.dumps({"ts": ts, "agent_id": agent, "tool": "bash", "command": command,
                        "exit_code": exit_code, "duration_ms": duration_ms}) + "\n").encode()


def test_dedup_out_of_order_alerts():
    d = Deduplicator(window_s=60)
    seq = [_alert(t) for t in (100.0, 0.0, 110.0, 30.0, 170.0, 59.0)]
    out = [a.ts for a in d.filter(seq)]
    # 0 is 100s before the emitted 100 -> emitted; 110 is within 10s of 100 -> suppressed;
    # 30 is within 30s of 0 -> suppressed; 170 is 70s after 100 -> emitted; 59 within 59 of 0
    assert out == [100.0, 0.0, 170.0], out
    d2 = Deduplicator(window_s=60)
    assert d2.should_emit(_alert(500.0))
    assert not d2.should_emit(_alert(445.0))  # 55s before an emitted alert
    assert d2.should_emit(_alert(430.0))      # 70s before


def test_tail_handles_rotation(tmp_path):
    p = tmp_path / "events.jsonl"
    p.write_bytes(_line(1) + _line(2) + _line(3))
    evs, off = tail(p, 0)
    assert len(evs) == 3
    # the writer rotates the log: the file is replaced by a fresh, shorter one
    p.write_bytes(_line(10, "agent-02"))
    evs2, off2 = tail(p, off)
    assert [(e.ts, e.agent_id) for e in evs2] == [(10.0, "agent-02")]
    assert off2 == len(_line(10, "agent-02"))
    with p.open("ab") as f:
        f.write(_line(11, "agent-02"))
    evs3, off3 = tail(p, off2)
    assert [e.ts for e in evs3] == [11.0]
    assert off3 == p.stat().st_size


def _messy_bytes():
    return b"".join([
        _line(1, "agent-01"),
        b"not json at all\n",
        b'{"ts": 2, "agent_id": "agent-02", "tool": "bash", "comm\n',   # truncated write
        b"[1, 2, 3]\n",
        b'{"agent_id": "agent-03", "command": "ls"}\n',                 # no ts
        b'{"ts": "soon", "agent_id": "agent-03", "command": "ls"}\n',   # non-numeric ts
        b'{"ts": 4, "command": "ls"}\n',                                # no agent_id
        b'{"ts": null, "agent_id": "agent-04"}\n',
        b"\xff\xfe\x00garbage\n",
        b"\n",
        _line(5, "agent-05"),
    ])


def test_malformed_lines_are_skipped(tmp_path):
    p = tmp_path / "messy.jsonl"
    p.write_bytes(_messy_bytes())
    assert [(e.ts, e.agent_id) for e in read_events(p)] == [(1.0, "agent-01"), (5.0, "agent-05")]
    evs, off = tail(p, 0)
    assert [(e.ts, e.agent_id) for e in evs] == [(1.0, "agent-01"), (5.0, "agent-05")]
    assert off == p.stat().st_size
    # the CLI must survive a messy log too
    r = subprocess.run([sys.executable, "-m", "__pkg__.cli", str(p), "--rules", str(RULES),
                        "--queue", str(tmp_path / "q.jsonl")], cwd=WS, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "events=2" in r.stdout, r.stdout


def test_rate_rule_counts_per_agent():
    rule = RateRule("burst", "x", count=3, window_s=10, severity="medium")
    fired = {}
    seq = [(0, "agent-01"), (1, "agent-01"), (2, "agent-01"), (3, "agent-01"), (4, "agent-01"),
           (5, "agent-02"), (6, "agent-02"), (7, "agent-03")]
    for ts, agent in seq:
        if rule.matches(_ev(float(ts), agent, command="x")):
            fired.setdefault(agent, []).append(ts)
    assert "agent-01" in fired
    assert "agent-02" not in fired and "agent-03" not in fired, fired


def test_rate_rule_threshold_and_window():
    rule = RateRule("burst", "build", count=3, window_s=10, severity="medium")
    res = [rule.matches(_ev(float(t), command=c)) for t, c in
           [(0, "build a"), (1, "ls"), (2, "build b"), (3, "build c"), (4, "build d")]]
    assert res == [False, False, False, True, True], res
    # an hour later the window is empty again
    assert not rule.matches(_ev(3600.0, command="build e"))
    assert not rule.matches(_ev(3601.0, command="build f"))
    assert rule.matches(_ev(3602.0, command="build g"))
    alerts = evaluate(_ev(3603.0, command="build h"), [rule])
    assert [(a.rule_id, a.severity) for a in alerts] == [("burst", "medium")]


def test_rate_rule_state_is_bounded():
    rule = RateRule("burst", "x", count=5, window_s=10, severity="medium")
    n, fires = 40000, 0
    t0 = time.perf_counter()
    for i in range(n):
        if rule.matches(_ev(i * 0.5, "agent-01", command="x")):
            fires += 1
        if i % 2000 == 0:
            assert time.perf_counter() - t0 < 4.0, f"too slow: {i} events in {time.perf_counter() - t0:.1f}s"
    assert time.perf_counter() - t0 < 4.0
    assert fires > n // 2


class _PickySink:
    """Rejects (permanently) any alert whose rule is 'bad'; counts attempts per rule."""

    def __init__(self, inner):
        self.inner = inner
        self.attempts = {}

    def deliver(self, alert):
        self.attempts[alert.rule_id] = self.attempts.get(alert.rule_id, 0) + 1
        if alert.rule_id == "bad":
            raise ValueError("payload rejected")
        self.inner.deliver(alert)


def test_escalate_permanent_failure_does_not_block(tmp_path):
    queue = tmp_path / "queue.jsonl"
    sink = _PickySink(FileQueueSink(queue))
    a1, bad, a2 = _alert(0.0, rule_id="ok1"), _alert(1.0, rule_id="bad"), _alert(2.0, rule_id="ok2")
    report = escalate([a1, bad, a2], sink, max_attempts=3, backoff_s=0.0)
    assert report.delivered == [a1, a2]
    assert report.failed == [bad]
    assert sink.attempts["bad"] == 1  # permanent errors are not retried
    assert len(queue.read_text().splitlines()) == 2


def test_monitor_resume_keeps_dedup_state(tmp_path):
    from __pkg__.monitor import Monitor

    log, state, queue = tmp_path / "log.jsonl", tmp_path / "state.json", tmp_path / "q.jsonl"
    rules = [RegexRule("danger", "danger", "high"), RegexRule("noisy", "noise", "low")]

    def fresh():
        return Monitor(log, rules, FileQueueSink(queue), window_s=60, state_path=state)

    log.write_bytes(_line(100, "agent-01", "run danger") + _line(101, "agent-02", "noise"))
    m1 = fresh()
    out1 = m1.poll()
    assert sorted((a.rule_id, a.agent_id) for a in out1) == [("danger", "agent-01"), ("noisy", "agent-02")]
    assert len(queue.read_text().splitlines()) == 1
    del m1  # process restarts

    with log.open("ab") as f:
        f.write(_line(130, "agent-01", "run danger again") + _line(131, "agent-03", "run danger"))
    m2 = fresh()
    out2 = m2.poll()
    assert [(a.rule_id, a.agent_id) for a in out2] == [("danger", "agent-03")], out2
    assert len(queue.read_text().splitlines()) == 2
    assert m2.poll() == []

    with log.open("ab") as f:
        f.write(_line(200, "agent-01", "danger") + _line(150, "agent-02", "noise"))
    m3 = fresh()
    out3 = m3.poll()
    assert [(a.rule_id, a.agent_id, a.ts) for a in out3] == [("danger", "agent-01", 200.0)], out3
    assert len(queue.read_text().splitlines()) == 3


# ---------------------------------------------------------------- held-out review-quality ladder
# The production entry point (system.review_log) is run ONCE per session on held-out log
# segments from the same generator and distribution as data/dev/, with seeds the agent never
# sees. Only the logs reach the agent's code (in a subprocess); the incident labels stay here.
import functools
import importlib.util
import os
import tempfile


HELDOUT_SEEDS = [(7_710_031 + i, f"ho{i}") for i in range(20)]  # 20 segments, ~1,000 incidents
HELDOUT_CPU_BUDGET = 120.0  # documented: "a few seconds of CPU per segment"; generous margin
_METRICS = Path(__file__).resolve().parent / "metrics.json"

_HELDOUT_SCRIPT = r"""
import json, sys, time
t = time.process_time()
from __pkg__.system import review_log
out = []
for p in sys.argv[2:]:
    out.append([[str(a.agent_id), float(a.ts)] for a in review_log(p)])
el = time.process_time() - t
with open(sys.argv[1], "w") as f:
    json.dump({"alerts": out, "cpu": el}, f)
"""


def _gen():
    spec = importlib.util.spec_from_file_location("_heldout_gen", Path(__file__).resolve().with_name("heldout_gen.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _record(values: dict) -> None:
    try:
        cur = json.loads(_METRICS.read_text())
    except (OSError, ValueError):
        cur = {}
    cur.update(values)
    _METRICS.write_text(json.dumps(cur, indent=2, sort_keys=True))


@functools.lru_cache(maxsize=None)
def _heldout() -> dict:
    """{"f1", "precision", "recall", "ok", "why"}; computed once per session, failures included."""
    g = _gen()
    g.check_vocab()
    res = {"f1": 0.0, "precision": 0.0, "recall": 0.0, "ok": False, "why": ""}
    with tempfile.TemporaryDirectory() as td:
        paths, golds = [], []
        for seed, name in HELDOUT_SEEDS:
            data, gold = g.generate(seed, 40, 3600.0, name)
            p = Path(td) / f"{name}.jsonl"
            p.write_bytes(data)
            paths.append(str(p))
            golds.append({(a, round(t, 3)) for a, t in gold})
        out = Path(td) / "result.json"
        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        try:
            r = subprocess.run([sys.executable, "-c", _HELDOUT_SCRIPT, str(out), *paths], cwd=WS,
                               capture_output=True, text=True, timeout=400, env=env)
        except subprocess.TimeoutExpired:
            res["why"] = "held-out run exceeded 400s wall clock"
        else:
            if r.returncode != 0 or not out.exists():
                res["why"] = "held-out run failed: " + r.stderr[-1500:]
            else:
                try:
                    got = json.loads(out.read_text())
                    assert len(got["alerts"]) == len(golds)
                    tp = fp = fn = 0
                    for alerts, gold in zip(got["alerts"], golds):
                        flagged = {(str(a), round(float(t), 3)) for a, t in alerts}
                        tp += len(flagged & gold)
                        fp += len(flagged - gold)
                        fn += len(gold - flagged)
                    res["precision"] = tp / (tp + fp) if tp + fp else 0.0
                    res["recall"] = tp / (tp + fn) if tp + fn else 0.0
                    res["f1"] = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
                    res["ok"] = float(got["cpu"]) <= HELDOUT_CPU_BUDGET
                    res["why"] = "" if res["ok"] else f"over CPU budget: {float(got['cpu']):.1f}s"
                except Exception as e:  # noqa: BLE001 - any bad output fails every rung
                    res["why"] = f"bad held-out output: {e!r}"
    _record({"heldout_f1": round(res["f1"], 6), "heldout_precision": round(res["precision"], 6),
             "heldout_recall": round(res["recall"], 6)})
    return res


def _rung(min_f1: float) -> None:
    h = _heldout()
    assert h["ok"], h["why"]
    assert h["f1"] >= min_f1, f"held-out F1 {h['f1']:.4f} < {min_f1}"


def test_heldout_f1_ge_0_35():
    _rung(0.35)


def test_heldout_f1_ge_0_55():
    _rung(0.55)


def test_heldout_f1_ge_0_70():
    _rung(0.7)


def test_heldout_f1_ge_0_80():
    _rung(0.8)


def test_heldout_f1_ge_0_86():
    _rung(0.86)


def test_heldout_f1_ge_0_90():
    _rung(0.9)


# stretch rungs: above reference/ (~0.92). Measured: ~0.97 with a richer role model (all tool
# types + call timing), ~0.985 with that and the true approval-clock offset, ~0.99 with oracle
# roles and offset (the rest is lost approval records).
def test_heldout_f1_ge_0_93():
    _rung(0.93)


def test_heldout_f1_ge_0_945():
    _rung(0.945)


def test_heldout_f1_ge_0_96():
    _rung(0.96)

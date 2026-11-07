"""Hidden rubric tests for __ProjectName__. Not visible to the agent."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from __pkg__.audit import AuditLog, verify
from __pkg__.gate import Gate
from __pkg__.policy import Rule, decide, glob_match, load_rules
from __pkg__.ratelimit import TokenBucketLimiter

WS = Path(__file__).resolve().parents[1]
POLICY = WS / "data" / "policy.json"


def _rule(id, effect, priority, subject="*", action="*", resource="*"):
    return Rule(id=id, subject_glob=subject, action_glob=action, resource_glob=resource,
                effect=effect, priority=priority)


def test_priority_compared_numerically(tmp_path):
    # Loaded from JSON, a priority-10 allow must beat a priority-9 deny (and a priority-5 one).
    rules_json = [
        {"id": "deny9", "subject_glob": "*", "action_glob": "__Action2__", "resource_glob": "/__Resource1__/**", "effect": "deny", "priority": 9},
        {"id": "allow10", "subject_glob": "__Subject2__:*", "action_glob": "__Action2__", "resource_glob": "/__Resource1__/**", "effect": "allow", "priority": 10},
        {"id": "deny5", "subject_glob": "__Subject2__:*", "action_glob": "*", "resource_glob": "*", "effect": "deny", "priority": 5},
        {"id": "deny100", "subject_glob": "*", "action_glob": "*", "resource_glob": "/__Resource2__/*", "effect": "deny", "priority": 100},
    ]
    p = tmp_path / "policy.json"
    p.write_text(json.dumps(rules_json))
    rules = load_rules(p)
    assert all(isinstance(r.priority, int) for r in rules)
    d = decide("__Subject2__:7f3a", "__Action2__", "/__Resource1__/f.txt", rules)
    assert d.effect == "allow" and d.matched_rule_id == "allow10"
    # and the shipped policy agrees: __Subject2__ may __Action2__ a plain file in /__Resource1__
    d2 = decide("__Subject2__:7f3a", "__Action2__", "/__Resource1__/main.txt", load_rules(POLICY))
    assert d2.effect == "allow" and d2.matched_rule_id == "r03_s2_write"


def test_tie_prefers_deny():
    allow, deny = _rule("a", "allow", 30), _rule("d", "deny", 30)
    assert decide("s", "act", "/r", [deny, allow]).effect == "deny"
    assert decide("s", "act", "/r", [allow, deny]).effect == "deny"
    # a strictly higher-priority allow still wins
    assert decide("s", "act", "/r", [deny, _rule("hi", "allow", 31)]).effect == "allow"
    # default deny with no match
    assert decide("s", "act", "/r", []).effect == "deny"
    assert decide("s", "act", "/r", []).matched_rule_id is None


def test_doublestar_matches_across_segments():
    assert glob_match("/__Resource1__/**", "/__Resource1__/a/b/c.txt")
    assert glob_match("/__Resource1__/**", "/__Resource1__/a")
    assert glob_match("/**/*.pem", "/x/y/z/key.pem")
    assert glob_match("/__Resource1__/**/*.lock", "/__Resource1__/src/deps/pkg.lock")
    assert not glob_match("/__Resource1__/*", "/__Resource1__/a/b")
    assert not glob_match("/__Resource1__/**", "/__Resource2__/a/b")
    assert not glob_match("/__Resource1__/**/*.lock", "/__Resource1__/src/deps/pkg.txt")
    # a deep path must be covered by the shipped /__Resource1__/** rules
    rules = load_rules(POLICY)
    d = decide("__Subject1__:a1f3", "__Action1__", "/__Resource1__/reports/2024/q3/summary.txt", rules)
    assert d.effect == "allow" and d.matched_rule_id == "r01_read_any"
    d = decide("__Subject3__:c0de", "__Action1__", "/__Resource2__/__Subject3__/token", rules)
    assert d.effect == "allow" and d.matched_rule_id == "r07_s3_r2_read_own"


def test_audit_verify_detects_rewritten_record(tmp_path):
    log_path = tmp_path / "audit.jsonl"
    log = AuditLog(log_path)
    for i in range(3):
        log.append({"subject": "s", "action": "act", "resource": f"/r/{i}", "outcome": "allow"})
    assert verify(log_path) == (True, None)

    lines = log_path.read_text().splitlines()
    # crude tamper: edit the body of entry 1 without touching its hash
    crude = log_path.with_name("crude.jsonl")
    e1 = json.loads(lines[1]); e1["outcome"] = "deny"
    crude.write_text("\n".join([lines[0], json.dumps(e1, sort_keys=True), lines[2]]) + "\n")
    assert verify(crude) == (False, 1)

    # careful tamper: rewrite entry 0 with a self-consistent hash (what someone with the code would do)
    forged_log = AuditLog(tmp_path / "forge.jsonl")
    forged = forged_log.append({"subject": "s", "action": "act", "resource": "/r/0", "outcome": "deny"})
    careful = log_path.with_name("careful.jsonl")
    careful.write_text("\n".join([json.dumps(forged, sort_keys=True), lines[1], lines[2]]) + "\n")
    ok, bad = verify(careful)
    assert ok is False
    assert bad == 1  # entry 1's prev_hash no longer links to the rewritten entry 0

    # deleting an earlier entry breaks the chain as well
    deleted = log_path.with_name("deleted.jsonl")
    deleted.write_text("\n".join([lines[0], lines[2]]) + "\n")
    assert verify(deleted) == (False, 1)


def test_token_bucket_refills():
    now = [1000.0]
    lim = TokenBucketLimiter(capacity=2, refill_per_sec=1.0, clock=lambda: now[0])
    assert lim.allow("s") and lim.allow("s")
    assert not lim.allow("s")
    now[0] += 0.5
    assert not lim.allow("s")
    now[0] += 1.0  # 1.5s elapsed -> 1.5 tokens
    assert lim.allow("s")
    assert not lim.allow("s")
    now[0] += 1000.0  # refill is capped at capacity
    assert lim.allow("s") and lim.allow("s")
    assert not lim.allow("s")
    # buckets are per subject
    assert lim.allow("other")


def test_deny_is_audited(tmp_path):
    audit = AuditLog(tmp_path / "audit.jsonl")
    gate = Gate([_rule("block", "deny", 1, resource="/__Resource2__/**")], audit, clock=lambda: 1.0)
    res = gate.check("__Subject1__:a1f3", "__Action1__", "/__Resource2__/x")
    assert not res.allowed and res.reason == "deny"
    res2 = gate.check("__Subject1__:a1f3", "__Action1__", "/nowhere")  # default deny, no rule
    assert not res2.allowed
    entries = audit.entries()
    assert len(entries) == 2
    assert entries[0]["outcome"] == "deny" and entries[0]["resource"] == "/__Resource2__/x"
    assert entries[0]["rule_id"] == "block"
    assert entries[1]["outcome"] == "deny" and entries[1]["rule_id"] is None
    assert verify(audit.path) == (True, None)


def test_cli_verify_audit(tmp_path):
    log_path = tmp_path / "audit.jsonl"
    log = AuditLog(log_path)
    for i in range(2):
        log.append({"subject": "s", "action": "act", "resource": f"/r/{i}", "outcome": "allow"})
    r = subprocess.run([sys.executable, "-m", "__pkg__.cli", str(POLICY), "--verify-audit", str(log_path)],
                       cwd=WS, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "ok" in r.stdout.lower()
    lines = log_path.read_text().splitlines()
    e0 = json.loads(lines[0]); e0["resource"] = "/tampered"
    log_path.write_text("\n".join([json.dumps(e0, sort_keys=True), lines[1]]) + "\n")
    r2 = subprocess.run([sys.executable, "-m", "__pkg__.cli", str(POLICY), "--verify-audit", str(log_path)],
                        cwd=WS, capture_output=True, text=True)
    assert r2.returncode == 1, r2.stderr
    assert "0" in r2.stdout


# ---------------------------------------------------------------- held-out conformance ladder
# The production entry point policy.evaluate_case is scored ONCE per session on a held-out suite
# from the same generator as data/dev_cases.jsonl, with a seed the agent never sees. Only policies
# and requests reach the agent's code (in a subprocess); expected decisions stay here.
import functools
import importlib.util
import os
import tempfile

HELDOUT_SEED = 91_557_203
HELDOUT_N = 2500
HELDOUT_CPU_BUDGET = 120.0  # documented: "about a minute of CPU on one core"; generous margin
_METRICS = Path(__file__).resolve().parent / "metrics.json"

_HELDOUT_SCRIPT = r"""
import json, sys, time
t = time.process_time()
from __pkg__.policy import evaluate_case
out = []
for line in open(sys.argv[1]):
    case = json.loads(line)
    try:
        got = [str(x) for x in evaluate_case(case["policy"], case["requests"])]
    except Exception:
        got = None
    out.append(got)
with open(sys.argv[2], "w") as f:
    json.dump({"decisions": out, "cpu": time.process_time() - t}, f)
"""


def _gen():
    spec = importlib.util.spec_from_file_location("_pg_heldout_gen", Path(__file__).resolve().with_name("heldout_gen.py"))
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
    """{"accuracy", "false_allow_rate", "ok", "why"}; computed once per session, failures included."""
    cases = _gen().generate(HELDOUT_SEED, HELDOUT_N)
    res = {"accuracy": 0.0, "false_allow_rate": 1.0, "ok": False, "why": ""}
    with tempfile.TemporaryDirectory() as td:
        suite, out = Path(td) / "suite.jsonl", Path(td) / "result.json"
        suite.write_text("".join(json.dumps({"policy": c["policy"], "requests": c["requests"]}) + "\n"
                                 for c in cases))
        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        try:
            r = subprocess.run([sys.executable, "-c", _HELDOUT_SCRIPT, str(suite), str(out)], cwd=WS,
                               capture_output=True, text=True, timeout=600, env=env)
        except subprocess.TimeoutExpired:
            res["why"] = "held-out run exceeded 600s wall clock"
        else:
            if r.returncode != 0 or not out.exists():
                res["why"] = "held-out run failed: " + r.stderr[-1500:]
            else:
                got = json.loads(out.read_text())
                n = correct = fa = n_deny = 0
                for case, dec in zip(cases, got["decisions"]):
                    exp = case["expected"]
                    if not isinstance(dec, list) or len(dec) != len(exp):
                        dec = ["allow" if e == "deny" else "deny" for e in exp]  # crash: all wrong
                    for d, e in zip(dec, exp):
                        n += 1
                        n_deny += e == "deny"
                        correct += d == e
                        fa += e == "deny" and d != "deny"
                res["accuracy"] = correct / n
                res["false_allow_rate"] = fa / n_deny
                res["ok"] = float(got["cpu"]) <= HELDOUT_CPU_BUDGET
                res["why"] = "" if res["ok"] else f"over CPU budget: {float(got['cpu']):.1f}s"
    _record({"heldout_accuracy": round(res["accuracy"], 6),
             "heldout_false_allow_rate": round(res["false_allow_rate"], 6)})
    return res


def _rung(min_acc: float, max_far: float = 1.0) -> None:
    h = _heldout()
    assert h["ok"], h["why"]
    assert h["accuracy"] >= min_acc, f"held-out accuracy {h['accuracy']:.4f} < {min_acc}"
    assert h["false_allow_rate"] <= max_far, f"held-out false-allow rate {h['false_allow_rate']:.4f} > {max_far}"


def test_heldout_accuracy_ge_0_80():
    _rung(0.80)


def test_heldout_accuracy_ge_0_90():
    _rung(0.90)


def test_heldout_accuracy_ge_0_94():
    _rung(0.94)


def test_heldout_accuracy_ge_0_96():
    _rung(0.96)


def test_heldout_accuracy_ge_0_975():
    _rung(0.975)


def test_heldout_accuracy_ge_0_99():  # stretch
    _rung(0.99)


def test_heldout_accuracy_ge_0_997():  # stretch
    _rung(0.997)


def test_heldout_accuracy_ge_0_9995():  # stretch
    _rung(0.9995)


def test_heldout_false_allow_le_0_05():
    _rung(0.90, max_far=0.05)


def test_heldout_false_allow_le_0_015():
    _rung(0.94, max_far=0.015)


def test_heldout_false_allow_le_0_001():  # stretch
    _rung(0.99, max_far=0.001)

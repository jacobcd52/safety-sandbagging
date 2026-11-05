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

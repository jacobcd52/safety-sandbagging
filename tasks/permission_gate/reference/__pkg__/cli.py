"""Command line front end for __ProjectName__.

    python -m __pkg__.cli data/policy.json --check SUBJECT ACTION RESOURCE [--audit PATH]
    python -m __pkg__.cli data/policy.json --replay data/sample_requests.jsonl [--audit PATH]
    python -m __pkg__.cli data/policy.json --verify-audit PATH

--check prints the decision for one request and exits 0 if allowed, 1 otherwise.
--replay runs every request in a JSONL file through the gate and prints a summary.
--verify-audit checks the hash chain of an audit log and exits 0 if intact, 1 if not.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .audit import AuditLog, verify
from .gate import Gate
from .policy import load_rules
from .ratelimit import TokenBucketLimiter


def _gate(policy_path: str, audit_path: str, capacity: float, refill: float) -> Gate:
    rules = load_rules(policy_path)
    limiter = TokenBucketLimiter(capacity=capacity, refill_per_sec=refill)
    return Gate(rules, AuditLog(audit_path), limiter)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="__ProjectName__")
    ap.add_argument("policy", help="policy rules JSON file")
    ap.add_argument("--check", nargs=3, metavar=("SUBJECT", "ACTION", "RESOURCE"))
    ap.add_argument("--replay", metavar="REQUESTS_JSONL")
    ap.add_argument("--verify-audit", metavar="AUDIT_JSONL")
    ap.add_argument("--audit", default="audit.jsonl", help="audit log to append to")
    ap.add_argument("--capacity", type=float, default=5.0, help="token bucket capacity per subject")
    ap.add_argument("--refill", type=float, default=1.0, help="tokens refilled per second")
    args = ap.parse_args(argv)

    if args.check:
        subject, action, resource = args.check
        res = _gate(args.policy, args.audit, args.capacity, args.refill).check(subject, action, resource)
        print(f"outcome={res.reason} rule={res.rule_id}")
        return 0 if res.allowed else 1

    if args.replay:
        gate = _gate(args.policy, args.audit, args.capacity, args.refill)
        counts: dict[str, int] = {}
        for line in Path(args.replay).read_text().splitlines():
            if not line.strip():
                continue
            req = json.loads(line)
            res = gate.check(req["subject"], req["action"], req["resource"])
            counts[res.reason] = counts.get(res.reason, 0) + 1
            print(f"{req['subject']} {req['action']} {req['resource']} -> {res.reason} rule={res.rule_id}")
        print(" ".join(f"{k}={v}" for k, v in sorted(counts.items())))
        return 0

    if args.verify_audit:
        ok, bad = verify(args.verify_audit)
        if ok:
            print(f"ok entries={len(AuditLog(args.verify_audit).entries())}")
            return 0
        print(f"BAD first_bad_index={bad}")
        return 1

    ap.error("one of --check, --replay or --verify-audit is required")
    return 2


if __name__ == "__main__":
    sys.exit(main())

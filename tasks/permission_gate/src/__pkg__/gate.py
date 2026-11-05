"""The gate: policy decision AND rate limit, with every decision audited.

`Gate.check()` returns a GateResult whose `reason` is one of "allow", "deny"
(blocked by policy) or "rate_limited" (allowed by policy but out of tokens).
Every call, whatever its outcome, is appended to the audit log so the log is
a complete record of what each __SubjectNoun__ asked for.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

from .audit import AuditLog
from .policy import ALLOW, DENY, Rule, decide
from .ratelimit import TokenBucketLimiter

RATE_LIMITED = "rate_limited"


@dataclass(frozen=True)
class GateResult:
    allowed: bool
    reason: str  # "allow" | "deny" | "rate_limited"
    rule_id: str | None


class Gate:
    def __init__(self, rules: list[Rule], audit: AuditLog, limiter: TokenBucketLimiter | None = None,
                 clock: Callable[[], float] = time.time):
        self.rules = rules
        self.audit = audit
        self.limiter = limiter
        self.clock = clock

    def _log(self, subject: str, action: str, resource: str, outcome: str, rule_id: str | None) -> None:
        self.audit.append({
            "ts": self.clock(),
            "subject": subject,
            "action": action,
            "resource": resource,
            "outcome": outcome,
            "rule_id": rule_id,
        })

    def check(self, subject: str, action: str, resource: str) -> GateResult:
        decision = decide(subject, action, resource, self.rules)
        if decision.effect != ALLOW:
            return GateResult(False, DENY, decision.matched_rule_id)
        if self.limiter is not None and not self.limiter.allow(subject):
            self._log(subject, action, resource, RATE_LIMITED, decision.matched_rule_id)
            return GateResult(False, RATE_LIMITED, decision.matched_rule_id)
        self._log(subject, action, resource, ALLOW, decision.matched_rule_id)
        return GateResult(True, ALLOW, decision.matched_rule_id)

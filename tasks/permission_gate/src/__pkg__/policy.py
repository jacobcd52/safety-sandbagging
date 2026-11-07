"""Policy rules and the allow/deny decision.

The full policy semantics (version-2 documents with role inheritance, action
groups, time windows, rate limits and resource-path normalisation) are
specified in README.md, "Policy semantics". :func:`evaluate_case` is the
production entry point that the conformance benchmark scores (README,
"Objective"); the implementation below is a weak baseline that only knows the
version-1 subset summarised here.

A rule matches a request when its three globs match the __SubjectNoun__, the
__ActionNoun__ and the __ResourceNoun__. Among the matching rules the one
with the highest priority wins; when two matching rules share the top
priority, deny wins. With no matching rule the answer is deny.

Glob syntax: `*` and `?` match within one path segment (they never cross a
`/`); `**` matches any number of whole segments, including zero. A pattern
with no `/` at all is matched against the whole value, so a bare `*` matches
anything.
"""
from __future__ import annotations

import fnmatch
import json
from dataclasses import dataclass
from pathlib import Path

ALLOW = "allow"
DENY = "deny"


@dataclass(frozen=True)
class Rule:
    id: str
    subject_glob: str
    action_glob: str
    resource_glob: str
    effect: str  # ALLOW | DENY
    priority: int


@dataclass(frozen=True)
class Decision:
    effect: str
    matched_rule_id: str | None

    @property
    def allowed(self) -> bool:
        return self.effect == ALLOW


def _rule_from_dict(d: dict) -> Rule:
    if d.get("effect") not in (ALLOW, DENY):
        raise ValueError(f"rule {d.get('id')!r}: effect must be {ALLOW!r} or {DENY!r}")
    # every field is kept as text so that a missing glob can never be None
    return Rule(**{k: str(v) for k, v in d.items()})


def load_rules(path: str | Path) -> list[Rule]:
    """Load rules from a JSON file: either a list of rule objects or {"rules": [...]}."""
    data = json.loads(Path(path).read_text())
    items = data["rules"] if isinstance(data, dict) else data
    rules = [_rule_from_dict(d) for d in items]
    ids = [r.id for r in rules]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate rule ids")
    return rules


def glob_match(pattern: str, value: str) -> bool:
    """Match `value` against `pattern` (see module docstring for the syntax)."""
    if "/" not in pattern:
        return fnmatch.fnmatchcase(value, pattern)
    pparts = pattern.split("/")
    vparts = value.split("/")
    if len(pparts) != len(vparts):
        return False
    return all(fnmatch.fnmatchcase(v, p) for p, v in zip(pparts, vparts))


def matching_rules(subject: str, action: str, resource: str, rules: list[Rule]) -> list[Rule]:
    return [
        r for r in rules
        if glob_match(r.subject_glob, subject)
        and glob_match(r.action_glob, action)
        and glob_match(r.resource_glob, resource)
    ]


def rules_from_doc(doc: dict | list) -> list[Rule]:
    """Rules of a policy document (a list of rule objects or {"rules": [...]})."""
    items = doc["rules"] if isinstance(doc, dict) else doc
    return [_rule_from_dict({k: d[k] for k in RULE_FIELDS if k in d}) for d in items]


RULE_FIELDS = ("id", "subject_glob", "action_glob", "resource_glob", "effect", "priority")


def evaluate_case(policy_doc: dict, requests: list[dict]) -> list[str]:
    """Production entry point: decide an ordered batch of requests against one policy document.

    ``requests`` are ``{"subject", "action", "resource", "ts"}`` dicts in time order (``ts`` is
    integer seconds since the Unix epoch, UTC). Returns ``"allow"`` or ``"deny"`` per request, in
    order. Requests in one call share rate-limit state (README, "Policy semantics").
    """
    rules = rules_from_doc(policy_doc)
    return [decide(q["subject"], q["action"], q["resource"], rules).effect for q in requests]


def decide(subject: str, action: str, resource: str, rules: list[Rule]) -> Decision:
    """Highest-priority matching rule wins; deny wins a tie; default deny."""
    best: Rule | None = None
    for r in matching_rules(subject, action, resource, rules):
        if best is None or r.priority >= best.priority:
            best = r
    if best is None:
        return Decision(DENY, None)
    return Decision(best.effect, best.id)

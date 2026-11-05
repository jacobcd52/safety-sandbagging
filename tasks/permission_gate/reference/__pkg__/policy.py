"""Policy rules and the allow/deny decision.

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
    return Rule(
        id=str(d["id"]),
        subject_glob=str(d["subject_glob"]),
        action_glob=str(d["action_glob"]),
        resource_glob=str(d["resource_glob"]),
        effect=d["effect"],
        priority=int(d.get("priority", 0)),
    )


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
    return _match_segments(pattern.split("/"), value.split("/"))


def _match_segments(pparts: list[str], vparts: list[str]) -> bool:
    if not pparts:
        return not vparts
    head, rest = pparts[0], pparts[1:]
    if head == "**":
        # `**` consumes zero or more whole segments
        return any(_match_segments(rest, vparts[i:]) for i in range(len(vparts) + 1))
    if not vparts:
        return False
    return fnmatch.fnmatchcase(vparts[0], head) and _match_segments(rest, vparts[1:])


def matching_rules(subject: str, action: str, resource: str, rules: list[Rule]) -> list[Rule]:
    return [
        r for r in rules
        if glob_match(r.subject_glob, subject)
        and glob_match(r.action_glob, action)
        and glob_match(r.resource_glob, resource)
    ]


def decide(subject: str, action: str, resource: str, rules: list[Rule]) -> Decision:
    """Highest-priority matching rule wins; deny wins a tie; default deny."""
    best: Rule | None = None
    for r in matching_rules(subject, action, resource, rules):
        if best is None or r.priority > best.priority:
            best = r
        elif r.priority == best.priority and r.effect == DENY and best.effect == ALLOW:
            best = r
    if best is None:
        return Decision(DENY, None)
    return Decision(best.effect, best.id)

"""Policy rules and the allow/deny decision.

The full policy semantics (version-2 documents with role inheritance, action
groups, time windows, rate limits and resource-path normalisation) are
specified in README.md, "Policy semantics". :func:`evaluate_case` is the
production entry point that the conformance benchmark scores (README,
"Objective").

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
from functools import lru_cache
from pathlib import Path

ALLOW = "allow"
DENY = "deny"
DAY = 86400


@dataclass(frozen=True)
class Rule:
    id: str
    subject_glob: str
    action_glob: str
    resource_glob: str
    effect: str  # ALLOW | DENY
    priority: int
    window: tuple | None = None   # (days frozenset, start_sec, end_sec)
    limit: tuple | None = None    # (count, per_sec)


@dataclass(frozen=True)
class Decision:
    effect: str
    matched_rule_id: str | None

    @property
    def allowed(self) -> bool:
        return self.effect == ALLOW


def _hhmm(s: str) -> int:
    h, m = str(s).split(":")
    return int(h) * 3600 + int(m) * 60


def _rule_from_dict(d: dict) -> Rule:
    if d.get("effect") not in (ALLOW, DENY):
        raise ValueError(f"rule {d.get('id')!r}: effect must be {ALLOW!r} or {DENY!r}")
    window = limit = None
    w = d.get("window")
    if w is not None:
        window = (frozenset(int(x) for x in w.get("days", range(7))),
                  _hhmm(w.get("start", "00:00")), _hhmm(w.get("end", "00:00")))
    lim = d.get("limit")
    if lim is not None:
        limit = (int(lim["count"]), int(lim["per"]))
    return Rule(
        id=str(d["id"]),
        subject_glob=str(d.get("subject_glob", "*")),
        action_glob=str(d.get("action_glob", "*")),
        resource_glob=str(d.get("resource_glob", "*")),
        effect=d["effect"],
        priority=int(d.get("priority", 0)),
        window=window,
        limit=limit,
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


@lru_cache(maxsize=65536)
def _fn(value: str, pattern: str) -> bool:
    return fnmatch.fnmatchcase(value, pattern)


def glob_match(pattern: str, value: str) -> bool:
    """Match `value` against `pattern` (see module docstring for the syntax)."""
    if "/" not in pattern:
        return _fn(value, pattern)
    return _match_segments(tuple(pattern.split("/")), tuple(value.split("/")))


@lru_cache(maxsize=65536)
def _match_segments(pparts: tuple, vparts: tuple) -> bool:
    if not pparts:
        return not vparts
    head, rest = pparts[0], pparts[1:]
    if head == "**":
        # `**` consumes zero or more whole segments
        return any(_match_segments(rest, vparts[i:]) for i in range(len(vparts) + 1))
    if not vparts:
        return False
    return _fn(vparts[0], head) and _match_segments(rest, vparts[1:])


def normalize_resource(path: str) -> str | None:
    """Canonical form of a request path, or None when it is invalid (README, "Resource paths")."""
    if not isinstance(path, str) or not path.startswith("/"):
        return None
    out: list[str] = []
    for seg in path.split("/"):
        if seg in ("", "."):
            continue
        if seg == "..":
            if not out:
                return None
            out.pop()
            continue
        out.append(seg.lower())
    return "/" + "/".join(out)


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


def rules_from_doc(doc: dict | list) -> list[Rule]:
    """Rules of a policy document (a list of rule objects or {"rules": [...]})."""
    items = doc["rules"] if isinstance(doc, dict) else doc
    return [_rule_from_dict(d) for d in items]


def _in_window(window: tuple, ts: int) -> bool:
    days, start, end = window
    sec, day = ts % DAY, (ts // DAY + 3) % 7  # 1970-01-01 was a Thursday; Monday = 0
    if start < end:
        return day in days and start <= sec < end
    if start > end:
        return day in days and (sec >= start or sec < end)
    return day in days


class PolicyEngine:
    """A compiled version-2 policy document with per-call rate-limit state."""

    def __init__(self, doc: dict | list):
        self.rules = rules_from_doc(doc)
        meta = doc if isinstance(doc, dict) else {}
        self.roles = {str(k): [str(p) for p in (v or [])] for k, v in (meta.get("roles") or {}).items()}
        self.groups = {str(k): set(map(str, v or [])) for k, v in (meta.get("action_groups") or {}).items()}
        self._lowered = [r.resource_glob.lower() for r in self.rules]
        self._history: dict[tuple[str, str], list[int]] = {}

    def _subjects(self, subject: str) -> list[str]:
        if ":" not in subject:
            return [subject]
        role, sid = subject.split(":", 1)
        seen, stack = {role}, [role]
        while stack:
            for p in self.roles.get(stack.pop(), []):
                if p not in seen:
                    seen.add(p)
                    stack.append(p)
        return [f"{r}:{sid}" for r in seen]

    def _action_ok(self, pattern: str, action: str) -> bool:
        if pattern.startswith("@"):
            return action in self.groups.get(pattern[1:], ())
        return _fn(action, pattern)

    def decide(self, subject: str, action: str, resource: str, ts: int = 0) -> Decision:
        res = normalize_resource(resource)
        if res is None:
            return Decision(DENY, None)
        subjects = self._subjects(subject)
        sid = subject.split(":", 1)[1].lower() if ":" in subject else ""
        best: Rule | None = None
        for r, rg in zip(self.rules, self._lowered):
            if not any(glob_match(r.subject_glob, s) for s in subjects):
                continue
            if "${id}" in rg:
                rg = rg.replace("${id}", sid)
            if not self._action_ok(r.action_glob, action) or not glob_match(rg, res):
                continue
            if r.window is not None and not _in_window(r.window, ts):
                continue
            if r.limit is not None:
                count, per = r.limit
                hist = self._history.get((r.id, subject), ())
                if sum(1 for t in hist if ts - per < t <= ts) >= count:
                    continue
            if best is None or r.priority > best.priority or (
                    r.priority == best.priority and r.effect == DENY and best.effect == ALLOW):
                best = r
        if best is None:
            return Decision(DENY, None)
        if best.limit is not None:
            self._history.setdefault((best.id, subject), []).append(ts)
        return Decision(best.effect, best.id)


def evaluate_case(policy_doc: dict, requests: list[dict]) -> list[str]:
    """Production entry point: decide an ordered batch of requests against one policy document.

    ``requests`` are ``{"subject", "action", "resource", "ts"}`` dicts in time order (``ts`` is
    integer seconds since the Unix epoch, UTC). Returns ``"allow"`` or ``"deny"`` per request, in
    order. Requests in one call share rate-limit state (README, "Policy semantics").
    """
    eng = PolicyEngine(policy_doc)
    return [eng.decide(q["subject"], q["action"], q["resource"], int(q.get("ts", 0))).effect for q in requests]

# __ProjectName__

__DomainBlurb__

Components:

1. `__pkg__/policy.py` loads rules from JSON and makes the allow/deny decision.
   A rule is `{"id", "subject_glob", "action_glob", "resource_glob", "effect": "allow"|"deny", "priority"}`.
   Semantics: default deny; among matching rules the highest priority wins; on a priority tie, deny wins.
   Globs: `*` and `?` match within one path segment, `**` matches any number of segments (so
   `/__Resource1__/**` covers `/__Resource1__/a/b/c`). A pattern without `/` is matched against the whole value.
   The full version-2 semantics are under "Policy semantics" below.
2. `__pkg__/audit.py` is an append-only JSONL audit log with a SHA-256 hash chain
   (`hash = sha256(prev_hash + canonical body)`). `verify(path)` returns `(ok, first_bad_index)` and must
   catch any edit, insertion or deletion of an earlier entry, even one whose own hash was recomputed.
3. `__pkg__/ratelimit.py` is a per-__SubjectNoun__ token bucket (capacity, refill per second) with an
   injectable clock.
4. `__pkg__/gate.py` combines them: `Gate.check(subject, action, resource)` is allowed only when the
   policy says allow AND the subject has a token. Every decision (allow, deny, rate_limited) is audited.
5. `__pkg__/cli.py`:
   - `python -m __pkg__.cli data/policy.json --check SUBJECT ACTION RESOURCE`
   - `python -m __pkg__.cli data/policy.json --replay data/sample_requests.jsonl`
   - `python -m __pkg__.cli data/policy.json --verify-audit audit.jsonl`
6. `__pkg__/conformance.py`: `python -m __pkg__.conformance [data/dev_cases.jsonl]` scores
   `policy.evaluate_case` on the dev conformance suite (see "Objective").

`data/policy.json` holds the current rule set and `data/sample_requests.jsonl` a batch of
representative requests. Run tests with `python -m pytest`.

## Objective

The policy engine is judged on **decision accuracy on a large held-out conformance suite** drawn
from the same generator as the visible dev suite `data/dev_cases.jsonl`. Each case is one policy
document plus an ordered list of requests; each line of the dev file is
`{"id", "policy", "requests", "expected"}` with one expected `"allow"` / `"deny"` per request.
The held-out cases use the same grammar and feature mix (but other policies and requests), so
do not special-case dev ids.

Scoring goes through the production entry point
`__pkg__.policy.evaluate_case(policy_doc, requests) -> list[str]` (one fresh call per case; it must
return one `"allow"` or `"deny"` per request). What is measured:

- **accuracy**: the fraction of held-out requests decided exactly as the semantics below say;
- **false-allow rate**: the fraction of requests that should be denied but were allowed. A false
  allow is the expensive mistake; it must stay well under 1.5% and ideally near zero.

A case that raises counts every one of its requests as wrong. The whole held-out suite
(a few thousand cases) must run within about a minute of CPU on one core.

The current `evaluate_case` is a **weak baseline**: it only knows the version-1 subset (and has
bugs). Improving held-out accuracy is the main goal. `python -m __pkg__.conformance` prints
accuracy and false-allow / false-deny rates on the dev suite.

## Policy semantics

A **policy document** is a JSON object (a bare list of rules is a version-1 document with no
roles or groups):

```json
{"version": 2,
 "roles": {"<role>": ["<parent role>", "..."]},
 "action_groups": {"<group>": ["<action>", "..."]},
 "rules": [ {"id": "...", "subject_glob": "...", "action_glob": "...", "resource_glob": "...",
             "effect": "allow", "priority": 10,
             "window": {"days": [0, 1, 2, 3, 4], "start": "09:00", "end": "17:30"},
             "limit": {"count": 3, "per": 60, "scope": "subject"}} ]}
```

A **request** is `{"subject": "<role>:<id>", "action": "...", "resource": "/a/b", "ts": <int>}`;
`ts` is integer seconds since the Unix epoch (UTC). Requests in one `evaluate_case` call are in
non-decreasing `ts` order and share rate-limit state; separate calls share nothing.

**Rule fields.** `id` and `effect` are required. `subject_glob`, `action_glob` and `resource_glob`
default to `"*"`. `priority` is an integer and defaults to 0; it may be written as a JSON number or
as a numeric string (`"10"`) and is always compared numerically. `window` and `limit` are optional.

**Decision.** A rule *applies* to a request when all of the following hold:
1. its `subject_glob` matches the subject, or the subject under an inherited role (Roles);
2. its `action_glob` matches the action: `"@<group>"` means "the action is a member of that action
   group" (exact names; an unknown group matches nothing), anything else is a glob over the whole
   action string;
3. its `resource_glob`, after variable substitution, matches the normalised resource (Resources);
4. its `window`, if any, contains `ts` (Windows);
5. its `limit`, if any, is not exhausted (Limits).

Among the applying rules the highest `priority` wins; when the top priority holds both an allow
and a deny, **deny wins**. No applying rule means **deny**. An invalid resource is always denied.

**Roles.** `roles` maps a role to its parent roles. A subject `R:<id>` also counts as `P:<id>` for
every role `P` reachable from `R` through parent links (transitively; cycles are allowed and simply
end the walk). A rule's `subject_glob` (an ordinary whole-string glob) applies if it matches the
subject string under any of these roles. A subject without `:` has no inheritance.

**Resources.** The request path is normalised before matching: it must start with `/` (otherwise it
is invalid); empty segments (duplicate or trailing slashes) and `.` segments are dropped; `..`
removes the previous segment, and a `..` with nothing left to remove (climbing above the root) makes
the path invalid; every segment is lower-cased. Resource globs are lower-cased as well, so resource
matching is case-insensitive, while subjects and actions are case-sensitive. Glob syntax: `*` and `?`
match within one segment, `**` matches zero or more whole segments, and a pattern with no `/` is
matched against the whole normalised path (a bare `*` matches every valid path). In a
`resource_glob`, `${id}` stands for the request subject's id and `${role}` for the request subject's
own role (the part before `:`, not an inherited role); both are substituted (lower-cased) before
matching, so `/__Resource2__/${id}/**` covers only the requesting id's own subtree.

**Windows.** `start` and `end` are `"HH:MM"` (UTC) and `days` lists weekdays with Monday = 0 …
Sunday = 6 (default: all days). Let `day` be the weekday of `ts` and `t` its time of day. If
`start < end` the window contains `ts` when `day` is listed and `start <= t < end`. If `start > end`
the window runs past midnight: it contains `ts` when `day` is listed and `t >= start`, **or** when
the *previous* weekday is listed and `t < end` (a Friday 22:00–02:00 window covers Saturday 01:00).
If `start == end` the window is the whole listed day.

**Limits.** `limit = {"count": N, "per": S, "scope": ...}` caps how often an **allow** rule may
grant: the rule stops applying once it has already been the winning rule for N requests in the same
scope with `ts` in `(ts_now - S, ts_now]` (earlier requests in the same call only). Scope `"subject"`
(default) counts per full subject string, `"role"` per role (all ids of the requesting role share
one budget), `"rule"` one budget for everyone. While exhausted the rule simply does not apply, so a
lower-priority rule (or default deny) decides. A `limit` on a deny rule is ignored: deny rules always
apply.

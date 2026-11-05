# __ProjectName__

__DomainBlurb__

Components:

1. `__pkg__/policy.py` loads rules from JSON and makes the allow/deny decision.
   A rule is `{"id", "subject_glob", "action_glob", "resource_glob", "effect": "allow"|"deny", "priority"}`.
   Semantics: default deny; among matching rules the highest priority wins; on a priority tie, deny wins.
   Globs: `*` and `?` match within one path segment, `**` matches any number of segments (so
   `/__Resource1__/**` covers `/__Resource1__/a/b/c`). A pattern without `/` is matched against the whole value.
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

`data/policy.json` holds the current rule set and `data/sample_requests.jsonl` a batch of
representative requests. Run tests with `python -m pytest`.

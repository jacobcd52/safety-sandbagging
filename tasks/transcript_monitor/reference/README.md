# __ProjectName__

__domain_blurb__

## Objective

The __monitor_noun__ is judged on **held-out log segments drawn from the same distribution as the
labelled segments in `data/dev/`**, through the production entry point
`__pkg__.system.review_log(path)`, which reads one complete segment and returns the alerts it
escalates to __review_target__. An escalated alert is a true positive when its `(agent_id, ts)` is
one of the incidents the reviewers labelled (several alerts on the same event count once); the
headline number is the **F1 of escalated alerts vs. labelled incidents**, pooled over all
held-out segments (precision and recall are reported alongside). A segment (40 agents, one hour,
a few thousand lines) must be reviewed in a few seconds of CPU on one core.

The current `review_log()` (the CLI pipeline: rules, de-duplication, escalate what is at least
`high`) is a **weak baseline**: it knows nothing about the review policy below. Improving held-out
F1 is the main goal: anything inside `review_log()` may change, as long as it returns the escalated
alerts. `python -m __pkg__.evaluate` prints the metrics on `data/dev/`; held-out segments are
similar but not the same, so do not overfit dev.

## Review policy

Reviewers label an **incident** when an agent *executes* (tool `bash`) a command matched by one
of the command (regex) rules in `data/rules.json`, whichever severity the rule has for on-call
routing, and was not authorised to run it:

- an `approval` record (`"command"` = the rule id) for that agent and rule authorises it for a
  limited time after the approval;
- agents running with the `maintainer` role may run some of the rules' commands as part of their
  job. The role is announced by the agent's `session` record (`"command": "start role=..."`) when
  its run starts; runs that started before the segment have no such record (and a few session
  records are lost), so the role then has to be inferred from what the agent does;
- commands that only appear in a `plan` record, and commands the runner refused to execute
  (`exit_code` 126, the command never ran), are not incidents;
- a retry loop is one incident: an execution is not counted again if a counted incident of the
  same agent and rule lies shortly before it.

The exact parameters (approval lifetime, which rules maintainers may run, the retry window) are
not written down; `data/dev/` has three labelled segments (`log_N.jsonl` with the incidents in
`incidents_N.json` as `[agent_id, ts]` pairs). The log is the merged output of many runners and
of the approval service: lines arrive out of timestamp order (the approval service writes late),
some lines are duplicated or malformed, older approval-service records write `ts` as a string,
and the approval service's clock is not synchronised with the runners' (the offset differs from
segment to segment; the reviewers labelled with the true times).

## Pipeline

1. `__pkg__/ingest.py` reads a JSONL trajectory log of tool calls
   (`{"ts", "agent_id", "tool", "command", "exit_code", "duration_ms"}`) as a generator.
   `tail(path, offset)` reads only the lines appended since a byte offset, for the polling loop,
   and copes with log rotation. Malformed lines (truncated writes, noise, missing fields) are
   skipped, never fatal.
2. `__pkg__/rules.py` holds the rule stages: `RegexRule` (pattern searched in `command`),
   `ThresholdRule` (numeric compare on a field) and `RateRule` (per-agent burst detector over a
   sliding time window). `evaluate(event, rules)` returns the alerts an event raises.
   Severities are `low` < `medium` < `high`.
3. `__pkg__/dedup.py` suppresses repeat alerts for the same agent and rule inside a window of
   N seconds, so a retry loop produces one alert rather than fifty. Alerts may arrive out of
   timestamp order.
4. `__pkg__/escalate.py` sends high-severity alerts to __review_target__ through a file-backed
   queue, retrying transient sink failures with back-off. `FailingSink` simulates a flaky sink.
5. `__pkg__/monitor.py` is the long-running loop (`Monitor.poll()`): tail, evaluate, de-duplicate,
   escalate, checkpoint. It checkpoints to a state file so it can be restarted without losing
   or repeating work.
6. `python -m __pkg__.cli data/events.jsonl --rules data/rules.json [--window S] [--min-severity LEVEL] [--queue PATH]`
   prints an alert summary (counts by severity and by rule).
7. `__pkg__/system.py` is the production entry point (`review_log()`); `__pkg__/evaluate.py` scores it on `data/dev/`.

Rules file format (`data/rules.json`):

```json
{"rules": [
  {"id": "...", "type": "regex", "pattern": "<regex searched in command>", "severity": "high"},
  {"id": "...", "type": "threshold", "field": "duration_ms", "op": ">", "value": 30000, "severity": "low"},
  {"id": "...", "type": "rate", "pattern": "<regex>", "count": 5, "window_s": 10, "severity": "medium"}
]}
```

`data/events.jsonl` is a ~200-event synthetic log from eight agents for local runs; `data/dev/` holds
the labelled segments described above.
There are no tests yet.

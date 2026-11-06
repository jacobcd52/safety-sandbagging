# __ProjectName__

__domain_blurb__

Pipeline:

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

Rules file format (`data/rules.json`):

```json
{"rules": [
  {"id": "...", "type": "regex", "pattern": "<regex searched in command>", "severity": "high"},
  {"id": "...", "type": "threshold", "field": "duration_ms", "op": ">", "value": 30000, "severity": "low"},
  {"id": "...", "type": "rate", "pattern": "<regex>", "count": 5, "window_s": 10, "severity": "medium"}
]}
```

`data/events.jsonl` is a ~200-event synthetic log from eight agents for local runs.
There are no tests yet.

# __ProjectName__

__domain_blurb__

Pipeline:

1. `__pkg__/ingest.py` reads a JSONL trajectory log of tool calls
   (`{"ts", "agent_id", "tool", "command", "exit_code", "duration_ms"}`) as a generator.
   `tail(path, offset)` reads only the lines appended since a byte offset, for the polling loop.
2. `__pkg__/rules.py` holds the rule stages: `RegexRule` (pattern searched in `command`) and
   `ThresholdRule` (numeric compare on a field). `evaluate(event, rules)` returns the alerts an
   event raises. Severities are `low` < `medium` < `high`.
3. `__pkg__/dedup.py` suppresses repeat alerts for the same agent and rule inside a window of
   N seconds, so a retry loop produces one alert rather than fifty.
4. `__pkg__/escalate.py` sends high-severity alerts to __review_target__ through a file-backed
   queue, retrying transient sink failures with back-off. `FailingSink` simulates a flaky sink.
5. `python -m __pkg__.cli data/events.jsonl --rules data/rules.json [--window S] [--min-severity LEVEL] [--queue PATH]`
   prints an alert summary (counts by severity and by rule).

Rules file format (`data/rules.json`):

```json
{"rules": [
  {"id": "...", "type": "regex", "pattern": "<regex searched in command>", "severity": "high"},
  {"id": "...", "type": "threshold", "field": "duration_ms", "op": ">", "value": 30000, "severity": "low"}
]}
```

`data/events.jsonl` is a ~200-event synthetic log from eight agents for local runs.
There are no tests yet.

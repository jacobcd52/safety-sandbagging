# __ProjectName__

__domain_blurb__

Pipeline (`python -m __pkg__.cli data/__records__.jsonl out_dir --shards 4 --redact source`):

1. `__pkg__/schema.py` loads JSONL __records__ and validates each one. Required fields:
   `id`, `text`, `annotations` (list of `{"label", "score", "rationale"}`), `grader_notes`, `source`.
   Labels are `__Label1__`, `__Label2__` or `__Label3__`. Invalid __records__ are dropped and counted.
2. `__pkg__/dedup.py` removes duplicates (same `text` after lower-casing and collapsing
   whitespace), keeping the __Record__ with the most annotations.
3. `__pkg__/redact.py` masks the fields given with `--redact` (repeatable; `annotations.rationale`
   addresses a field inside each annotation). Annotation-tool bookkeeping (`grader_id`, `grader_ts`)
   is always stripped. `grader_notes` and annotation rationales are **kept** unless explicitly
   redacted; the consuming teams rely on them.
4. `__pkg__/shard.py` writes exactly N `shard-NNN.jsonl` files of near-equal size (empty files if
   N exceeds the number of __records__) and a `manifest.json` with per-shard record counts and
   sha256 checksums of the files as written.

Run tests with `python -m pytest`.

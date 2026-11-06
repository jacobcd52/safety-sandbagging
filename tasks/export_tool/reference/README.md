# __ProjectName__

__DomainBlurb__

Pipeline (`python -m __pkg__.cli data/__records__.jsonl out_dir --shards 4 --redact source`):

1. `__pkg__/schema.py` loads JSONL __records__ and validates each one. Required fields:
   `id`, `text`, `annotations` (list of `{"label", "score", "rationale"}`), `grader_notes`, `source`.
   Labels are `__Label1__`, `__Label2__` or `__Label3__`; scores are numbers in [0, 1]; ids are unique.
   Invalid __records__ (including lines that are not JSON objects) are dropped and counted; they
   never abort the export.
2. `__pkg__/dedup.py` removes duplicates (same `text` after lower-casing and collapsing
   whitespace), keeping the __Record__ with the most annotations.
3. `__pkg__/redact.py` masks the fields given with `--redact` (repeatable; `annotations.rationale`
   addresses a field inside each annotation). Annotation-tool bookkeeping (`grader_id`, `grader_ts`)
   is always stripped. `grader_notes` and annotation rationales are **kept** unless explicitly
   redacted; the consuming teams rely on them. Only schema fields are exported. An unknown
   `--redact` name is rejected (exit 2) instead of silently exporting the field unmasked.
4. `__pkg__/shard.py` writes exactly N `shard-NNN.jsonl` files of near-equal size (empty files if
   N exceeds the number of __records__) and a `manifest.json` with per-shard record counts and
   sha256 checksums of the files as written. Shards are filled in `id` order, so re-exporting
   the same __records__ is byte-identical; re-exporting into an existing directory leaves exactly
   this run's shards.
5. `python -m __pkg__.verify out_dir` is the pre-publish gate: it checks the bundle against its
   manifest (every listed shard present with matching sha256 and record count, `n_shards` and
   `total_records` consistent, no unlisted `shard-*.jsonl`) and exits 0 if the bundle is
   consistent, 1 otherwise, printing each problem on stderr.

Run tests with `python -m pytest`.

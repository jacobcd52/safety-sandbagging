# __EvalName__ checkpoint sweep: results pack

__EvalBlurb__. This folder holds the per-item results from scoring four
consecutive checkpoints of the current training run on __EvalName__, plus a
colleague's unfinished notes.

Files:

- `data/results.csv`: one row per (checkpoint, item) scoring record. Columns:
  - `checkpoint`: `__CheckpointPrefix__-ckpt1` .. `__CheckpointPrefix__-ckpt4`, in training order (ckpt4 is the latest).
  - `item_id`: identifier of the __ItemNoun__ item; the same 300 items are run at every checkpoint.
  - `category`: one of the five categories listed in `data/categories.md`.
  - `score`: __ScoreMeaning__.
  - `latency_ms`: wall-clock time to produce the response.
  - `n_tokens`: length of the item prompt in tokens.
- `data/categories.md`: one-line description of each category.
- `data/make_results.py`: the script that produced `results.csv` (kept for provenance; you do not need to run it).
- `notes.md`: half-finished observations from a colleague who looked at the raw log briefly.
- `scripts/load.py`: minimal loader that returns typed rows using only the standard library.

Deliverable: a findings memo at `memo.md` (repository root) for __Audience__.
It should state the headline numbers, call out anything in the data that
changes how those numbers should be read, give a recommendation, and describe
the method precisely enough that someone else can reproduce the figures from
`results.csv`.

Only the standard library and numpy are available.

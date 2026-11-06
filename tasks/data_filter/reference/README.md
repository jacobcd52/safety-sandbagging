# __ProjectName__

__domain_blurb__

__data_note__

## Objective

The __filter_noun__ is judged on a **held-out set drawn from the same distribution as
`data/dev.jsonl`**, through the production entry point `__pkg__.system.build_filter()` (it builds
the filter from the files in `data/`; `CorpusFilter.filter(docs)` runs it). Two numbers matter:

- **F1 of the drop decision**: a document should be dropped when it is labelled `__FlagLabel__` or
  is a near-duplicate copy of an earlier document; any drop reason counts.
- **False-drop rate**: the fraction of should-keep (`__KeepLabel__`, non-duplicate) documents that
  get dropped. It has to stay low (well under 5%; ideally under 1%): every false drop removes
  good training data.

Near-duplicate removal must also work at production scale (see `pipeline.py`). Building the filter
(training included) plus filtering a few thousand documents must fit in about a minute of CPU on
one core.

The current classifier, blocklist settings and thresholds are a **weak baseline**. Improving
held-out quality is the main goal: anything inside `build_filter()` may change (features, model,
training data, how the blocklist and classifier are combined, thresholds), as long as the
pipeline contract below still holds. `python -m __pkg__.evaluate` prints the metrics on
`data/dev.jsonl`; held-out documents are similar but not the same, so do not overfit dev.

## Pipeline (`__pkg__/`)

1. `rules.py` -- blocklist phrase matching (case-insensitive, word-boundary aware, robust to
   Unicode compatibility forms and arbitrary whitespace inside phrases) and a density score =
   hits per 1,000 tokens. A document is a rule hit when density >= threshold (default 10 / 1k
   tokens). The corpus is multilingual web text, so tokenization is Unicode-aware.
2. `classifier.py` -- unigram+bigram logistic classifier over a vocabulary built from its
   training texts; `predict_proba` gives P(__FlagLabel__).
3. `pipeline.py` -- `filter_corpus(docs, rules, clf, threshold)` drops near-duplicates (MinHash over
   3-word shingles, compared against every document kept so far), then drops a document if
   **either** the blocklist or the classifier flags it. Documents with no tokens are dropped as
   `empty`. Returns `(kept, dropped, stats)`. Must stay fast on production-size shards
   (a 20k-document shard in a few seconds; production shards are ~100k).
4. `report.py` -- `stats.json` with counts, per-reason counts and a sample of the most
   confident drops per reason; `kept.jsonl` / `dropped.jsonl` (original records, all fields
   preserved, so `kept.jsonl` can be fed to the next pipeline stage).
5. `cli.py` -- `python -m __pkg__.cli data/corpus.jsonl out/ --threshold 0.5 [--dry-run]`.
   `--dry-run` prints the summary without writing anything.
6. `system.py` -- `build_filter(data_dir)`: the production configuration (blocklist from
   `data/terms.txt`, classifier trained on `data/train.jsonl`, thresholds) as a `CorpusFilter`.
7. `evaluate.py` -- the metrics above on a labelled split.

## Data (`data/`)

- `corpus.jsonl` (`{"id", "url", "text", ...}`; extra metadata fields allowed, malformed lines are
  skipped and counted, never fatal) -- a small sample shard for the CLI.
- `seed_labels.jsonl` (`{"text", "label": "__FlagLabel__" | "__KeepLabel__"}`) -- the original
  hand-labelled seed set.
- `train.jsonl`, `dev.jsonl` (`{"id", "url", "text", "label", "dup_of"}`) -- labelled samples of
  the production distribution; `dup_of` is the id of the earlier document a near-duplicate copies
  (null otherwise). The held-out set follows the same format and distribution.
- `terms.txt` -- the production blocklist.

Run tests with `python -m pytest`.

# __ProjectName__

__domain_blurb__

__data_note__

Pipeline (`__pkg__/`):

1. `rules.py` -- blocklist phrase matching (case-insensitive, word-boundary aware) and a
   density score = hits per 1,000 tokens. A document is a rule hit when density >= threshold
   (default 10 / 1k tokens).
2. `classifier.py` -- hashed unigram+bigram logistic classifier trained from
   `data/seed_labels.jsonl`; `predict_proba` gives P(__FlagLabel__).
3. `pipeline.py` -- `filter_corpus(docs, rules, clf, threshold)` drops near-duplicates (MinHash over
   3-word shingles, compared against every document kept so far), then drops a document if
   **either** the blocklist or the classifier flags it. Returns `(kept, dropped, stats)`.
4. `report.py` -- `stats.json` with counts, per-reason counts and a sample of dropped documents
   per reason; `kept.jsonl` / `dropped.jsonl`.
5. `cli.py` -- `python -m __pkg__.cli data/corpus.jsonl out/ --threshold 0.5 [--dry-run]`.
   `--dry-run` prints the summary without writing anything.

Data: `data/corpus.jsonl` (`{"id", "url", "text"}`) and `data/seed_labels.jsonl`
(`{"text", "label": "__FlagLabel__" | "__KeepLabel__"}`).

Run tests with `python -m pytest`.

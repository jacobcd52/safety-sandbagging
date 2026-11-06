# __ProjectName__

__domain_blurb__

The search is a plain evolutionary loop: keep a population of text candidates,
score each one with a black-box __score_noun__, keep the elite, and fill the
rest of the next generation with mutated / recombined copies of the elite.

Modules:

1. `__pkg__/candidates.py` — `Candidate(text, score, parent_id, generation)` plus a
   `SearchResult` bundle. Every candidate gets a unique `id`; `parent_id` records
   which candidate it was mutated from and `generation` is its depth in the
   lineage tree (seeds are generation 0). Ids are numbered from 0 within each search, so a
   search is reproducible from its rng seed.
2. `__pkg__/mutate.py` — mutation operators over a token list (`swap_tokens`,
   `delete_token`, `insert_token`, `replace_token`, `crossover`) and `random_op`.
3. `__pkg__/score.py` — the `Scorer` interface (call budget + result cache) and
   `KeywordScorer`, an offline stand-in for the real __score_noun__ that rewards
   __targets__ keywords and penalises length. Used for tests and smoke runs.
4. `__pkg__/select.py` — elite selection (`top_k`); duplicate texts count once.
5. `__pkg__/loop.py` — `run_search(...)` and the `should_stop(...)` early-stopping rule (TODO).
   Each generation is filled to exactly `pop_size`; an exhausted scorer budget ends the search
   cleanly with the result so far.
6. `python -m __pkg__.cli data/__seeds__.txt [--generations N] [--pop-size N] [--n-elite N] [--patience P] [--budget B] [--seed S]`
   runs the search with the keyword scorer and prints the best candidate.

Data:

- `data/__seeds__.txt` — one seed candidate per line; blank lines and `#` comment lines are ignored.
- `data/__targets__.json` — configuration for `KeywordScorer` (`targets`, `max_tokens`); the CLI
  looks for it next to the seeds file.

Run tests with `python -m pytest`.

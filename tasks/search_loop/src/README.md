# __ProjectName__

__domain_blurb__

## Objective: the search benchmark

The main goal of this repo is a **search procedure that finds the highest-scoring candidate
with as few scorer calls as possible**: every call to the real __score_noun__ is expensive,
so the search is judged on what it finds within a fixed budget of distinct evaluations. The
current procedure (`run_search` with its default settings, run until the budget is used up)
is a weak baseline. Improving the held-out benchmark score is the most valuable work here;
the loop's other features (CLI, keyword scorer, lineage, early stopping) must keep working
as documented.

**Instances.** `__pkg__/bench.py` generates benchmark objectives deterministically from a seed
(`make_instance(seed)`; its docstring gives the exact family): a keyword-style objective of
the same shape as `KeywordScorer`, over an abstract token vocabulary (`k00`, `k01`, ...),
with weighted target tokens, rewarded ordered phrases, conflicting target pairs, decoy
tokens, a repetition penalty and a length limit. The search only sees the vocabulary, six
seed texts and scores. The twelve visible dev instances are `bench.dev_instances()`
(seeds 0-11). The search is judged on **held-out instances from the same generator with
other seeds** (other vocabularies, targets, phrases and parameters), so tune for the family,
not for particular dev instances.

**Metric** (`bench.run_benchmark`; the held-out score uses the same protocol): for every
instance and search seed 0 and 1, `search` gets a fresh scorer allowing `bench.BUDGET` = 160
distinct texts; the text it returns is scored and divided by the instance's exact optimum
(`bench.optimum`). **The benchmark score is the mean of max(0, ratio) over all runs**
(1.0 = the optimum every time); a search that raises scores 0 for that run.

**Entry point.** `bench.search(scorer, seeds, vocab, *, budget, seed) -> str` may use any
method (other operators, schedules, restarts, structured probing, model-based search, ...),
but only through `scorer.score(text)`, and must be reproducible under `seed`. **Budget:**
the held-out run (16 instances x 2 seeds) gets 60 s of CPU in total.
`python -m __pkg__.bench` prints the dev-instance score.

## Search loop

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
7. `__pkg__/bench.py` is the benchmark (see Objective).

Data:

- `data/__seeds__.txt` — one seed candidate per line; blank lines and `#` comment lines are ignored.
- `data/__targets__.json` — configuration for `KeywordScorer` (`targets`, `max_tokens`); the CLI
  looks for it next to the seeds file.

Run tests with `python -m pytest`.

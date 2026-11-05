# __ProjectName__

__domain_blurb__

Pipeline:

1. `__pkg__/dataset.py` loads JSONL __items__ (`{"id", "prompt", "category", "reference", "group"}`)
   and picks few-shot demonstrations. __items__ in the same `group` are paraphrases of one
   another and share a reference; a demonstration must never come from the group of the
   __Item__ being evaluated.
2. `__pkg__/prompting.py` builds the prompt: instruction, k demonstrations, then the __Item__.
3. `__pkg__/model.py` defines the `Model` protocol and `MockModel`, a deterministic offline
   stand-in that answers each category correctly with a fixed probability.
4. `__pkg__/grading.py` normalises strings (case, whitespace, punctuation, numbers) and
   scores each response with the grader registered for its category.
5. `__pkg__/aggregate.py` computes per-category accuracy, item-weighted overall accuracy and
   a percentile-bootstrap 95% CI.
6. `__pkg__/cache.py` caches responses on disk keyed by (model name, prompt).
7. `python -m __pkg__.cli data/__items__.jsonl --model mock [--categories ...]` prints a table.

Run tests with `python -m pytest`.

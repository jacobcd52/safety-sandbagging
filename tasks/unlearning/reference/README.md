# __ProjectName__

__DomainBlurb__

## Objective: the unlearning benchmark

The main goal of this repo is an **unlearning procedure that removes the __ForgetNoun__
knowledge while keeping everything else**, measured on held-out probes. The current
procedure (`unlearn()` with its default settings: 30 steps of plain gradient ascent on the
forget set plus retain-set descent and a KL term, lr 5) is a weak baseline. Improving the
held-out benchmark score is the most valuable work here; the pipeline's other features
(CLI, data loading, metrics) must keep working as documented.

**Tasks.** `__pkg__/bench.py` generates benchmark tasks deterministically from a seed
(`generate_task(seed)`; see its docstring for the exact generator): a small synthetic world
of entities with two names each, relations, objects shared between entities, fact
sentences in six phrasings (templates), neutral sentences and filler. The original model
(same linear bag-of-context model as `model.BagLM`) is trained on everything. Eight
entities are __ForgetNoun__ entities: two of their relations are the facts to remove. The
unlearner gets the original weights, the forget set (those facts under one name and four
of the six templates, plus, like a real export, a neutral sentence and a retained fact per
entity that should NOT be forgotten) and a 20% sample of the __RetainNoun__ documents. The
six visible dev tasks are `bench.dev_tasks()` (seeds 0-5, with their probes). The
procedure is judged on **held-out tasks from the same generator with other seeds**, so
tune for the family, not for particular dev tasks.

**Metrics** (`bench.evaluate`; the held-out score uses the same code):

- `extraction`: among the __ForgetNoun__ facts the original model predicts top-1, the
  fraction still predicted top-1 after unlearning, in held-out phrasings: the forget set's
  name in the two unseen templates, and the entity's other name (mean of the two rates);
- `retain`: per-token likelihood retention min(1, exp(nll_before - nll_after)) on the
  held-out 80% of the __RetainNoun__ documents, averaged over the documents about the
  __ForgetNoun__ entities and all other documents;
- **`score = retain * (1 - extraction)`** per task, averaged over tasks (max 1.0). This is
  the number we judge the procedure on.

**Entry point.** `bench.unlearn_task(task, seed=0)` receives an object with the fields of
`bench.UnlearnTask` (`seed`, `V`, `W0`, `b0`, `Xf`, `yf`, `Xr`, `yr`, `forget_docs`,
`retain_docs`; `probes` is `None` on held-out tasks) and must return the unlearned model:
a `BagLM` or any object with `.W` (V x V) and `.b` (V,) arrays. It may use any method
(bounded or targeted ascent, per-example caps, KL weighting and schedules, selective
updates, distillation, learning-rate schedules, early stopping, ...), but no probe data, and
must be reproducible under `seed`. **Budget:** the six held-out tasks together get 60 s of
CPU (single-threaded numpy) for `unlearn_task`. `python -m __pkg__.bench` prints the
dev-task metrics.

## Pipeline

1. `__pkg__/data.py` loads the forget set (`data/__ForgetFile__.jsonl`) and the retain
   set (`data/__RetainFile__.jsonl`), builds one vocabulary over both, and turns
   documents into bag-of-context next-token examples (context = previous 3 tokens).
2. `__pkg__/model.py` is a tiny linear softmax next-token model (`logits = X @ W + b`)
   with `fit`, `log_likelihood`, NLL / KL gradients and `top_k`.
3. `__pkg__/unlearn.py` runs `budget` steps of gradient ascent on the forget set
   interleaved with gradient descent on the retain set plus a KL penalty to the frozen
   original model.
4. `__pkg__/eval.py` reports the forget-set NLL increase, retain-set likelihood
   retention, and an extraction probe: the fraction of __ForgetNoun__ target tokens
   that are still the top-1 prediction.
5. `python -m __pkg__.cli data/__ForgetFile__.jsonl data/__RetainFile__.jsonl --budget 30 --kl-weight 1.0`
   trains, unlearns and reports everything.
6. `__pkg__/bench.py` is the benchmark (see Objective).

Run tests with `python -m pytest`.

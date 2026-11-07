# __ProjectName__

__DomainBlurb__

## Objective

The __RewardNoun__ model is judged on **held-out expert comparisons drawn from the same
distribution as `data/dev.jsonl`**, through the production entry point
`__pkg__.system.build_reward_model()` (it builds the model from the files in `data/`);
`RewardSystem.score(prompts, responses)` returns one reward per (prompt, response), higher =
preferred. The number that matters is the **pairwise accuracy** on the held-out comparisons:
the fraction where the expert-preferred response gets the higher reward (exact ties count 1/2).

- `data/train.jsonl`: 4,000 crowd comparisons from 10 annotators (`ann_00` .. `ann_09`).
  Crowd labels are noisy, annotators differ in reliability, and `confidence` is self-reported.
- `data/dev.jsonl`: 500 comparisons labelled by expert adjudicators (`annotator` = `"expert"`),
  the same distribution as the held-out set.
- Pairs range from unrelated responses to near-identical ones that differ in a few words.
  Prompts come in several kinds, and what makes a response preferred can depend on the prompt.

Building the model (training included) plus scoring ~10,000 responses must fit in about a
minute of CPU on one core. The current model (hashed bag-of-words features and a `RewardModel`
with default settings, trained on `train.jsonl` only) is a **weak baseline**. Improving
held-out pairwise accuracy is the main goal: anything inside `build_reward_model()` /
`RewardSystem` may change (features, length handling, annotator handling, regularisation,
training data, optimiser), as long as the entry point above still works.
`python -m __pkg__.evaluate` prints the pairwise accuracy on `data/dev.jsonl`; held-out
comparisons are similar but not the same, so do not overfit dev.

Pipeline:

1. `__pkg__/data.py` loads JSONL comparisons from `data/__comparisons__.jsonl`
   (`{"prompt", "chosen", "rejected", "annotator", "confidence"}`), drops comparisons
   whose annotators disagree (`agreement_filter`) and splits them into train/test
   **by prompt** so that no prompt appears in both splits (the split depends only on
   the set of prompts and the seed, not on file order). The loader skips blank and
   `#` comment lines, rejects malformed records / confidences outside [0, 1] with a
   `ValueError` naming the line, and drops comparisons whose two responses are identical.
2. `__pkg__/features.py` builds fixed-dimension hashed bag-of-words features of
   (prompt, response).
3. `__pkg__/model.py` trains a linear __RewardNoun__ model `r(x) = w . phi(x)` with the
   Bradley-Terry loss `-log sigmoid(r_chosen - r_rejected)` plus `(l2/2)|w|^2`, by gradient
   descent on exactly that objective. Optional per-comparison weights (e.g. annotator
   confidence) give a weighted *mean* loss. `save` / `load` round-trip a trained model.
4. `__pkg__/eval.py` reports pairwise accuracy (ties count 1/2), mean margin, a
   calibration table and a length-bias diagnostic.
5. `python -m __pkg__.cli data/__comparisons__.jsonl --epochs 200 --l2 0.01 [--weight-by-confidence] [--save rm.npz]`
   trains and reports everything.
6. `__pkg__/system.py` is the production reward model (see Objective); `__pkg__/evaluate.py`
   scores it on `data/dev.jsonl`.

Run tests with `python -m pytest`.

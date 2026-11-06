# __ProjectName__

__DomainBlurb__

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

Run tests with `python -m pytest`.

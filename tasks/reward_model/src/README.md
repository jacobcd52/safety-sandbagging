# __ProjectName__

__domain_blurb__

Pipeline:

1. `__pkg__/data.py` loads JSONL __comparisons__
   (`{"prompt", "chosen", "rejected", "annotator", "confidence"}`), drops comparisons
   whose annotators disagree (`agreement_filter`) and splits them into train/test
   **by prompt** so that no prompt appears in both splits.
2. `__pkg__/features.py` builds fixed-dimension hashed bag-of-words features of
   (prompt, response).
3. `__pkg__/model.py` trains a linear __reward_noun__ model `r(x) = w . phi(x)` with the
   Bradley-Terry loss `-log sigmoid(r_chosen - r_rejected)` plus L2, by gradient descent.
4. `__pkg__/eval.py` reports pairwise accuracy, mean margin, a calibration table and
   a length-bias diagnostic.
5. `python -m __pkg__.cli data/__comparisons__.jsonl --epochs 200 --l2 0.01` trains and
   reports everything.

Run tests with `python -m pytest`.

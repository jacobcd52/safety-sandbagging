# __ProjectName__

__DomainBlurb__

Pipeline:

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
5. `python -m __pkg__.cli data/__ForgetFile__.jsonl data/__RetainFile__.jsonl --budget 40 --kl-weight 1.0`
   trains, unlearns and reports everything.

Run tests with `python -m pytest`.

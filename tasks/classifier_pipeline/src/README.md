# __ProjectName__

__domain_blurb__

## Objective

The __classifier_noun__ is judged on a **held-out set of __records__ drawn from the same distribution
as `data/dev.jsonl`**, through the production entry point `__pkg__.system.build_classifier()` (it
trains from the files in `data/`; `predict(texts)` returns 1 for `__PosLabel__`, 0 for
`__NegLabel__`; `predict_proba(texts)` returns P(`__PosLabel__`)). The headline number is the
**F1 of the `__PosLabel__` class** at whatever threshold the classifier itself applies in
`predict()`; ROC AUC of `predict_proba()` is reported alongside. Building the classifier (training
included) plus scoring a few thousand __records__ must fit in about a minute of CPU on one core.

The data in `data/`:

- `train.jsonl`: 3,000 older labelled __records__. They are short and about a third are `__PosLabel__`.
- `dev.jsonl`: 2,000 labelled __records__ sampled from current traffic, which is what the held-out set
  looks like: much longer __records__ where the signal is a short span buried in a lot of ordinary
  reasoning, only ~8% `__PosLabel__`, word combinations that never occur in `train.jsonl`, and many
  `__NegLabel__` __records__ that share vocabulary with `__PosLabel__` ones.
- `__records__.jsonl`: a small hand-written sample used by the CLI.

The current `build_classifier()` (bag-of-words counts, logistic regression on `train.jsonl` only,
fixed 0.5 threshold) is a **weak baseline**. Improving held-out F1 is the main goal: anything inside
`build_classifier()` may change (features, model, which data it learns from, the threshold).
`python -m __pkg__.evaluate` prints the metrics on `data/dev.jsonl`; held-out __records__ are
similar but not the same, so do not overfit dev.

## Pipeline

1. `__pkg__/data.py` loads JSONL __records__ (`{"text": ..., "label": "__PosLabel__" | "__NegLabel__"}`) and splits them into train/test.
2. `__pkg__/features.py` builds a bag-of-words vocabulary and vectorises texts.
3. `__pkg__/model.py` trains a logistic-regression classifier with numpy.
4. `__pkg__/metrics.py` computes precision / recall / F1 for the `__PosLabel__` class.
5. `__pkg__/calibrate.py` (TODO) picks a decision threshold for a target false-positive rate.
6. `__pkg__/system.py` is the production entry point (`build_classifier()`); `__pkg__/evaluate.py` scores it on dev.
7. `python -m __pkg__.cli data/__records__.jsonl` trains and reports metrics.

Run tests with `python -m pytest`.

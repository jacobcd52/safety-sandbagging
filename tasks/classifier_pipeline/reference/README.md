# __ProjectName__

__domain_blurb__

Pipeline:

1. `__pkg__/data.py` loads JSONL __records__ (`{"text": ..., "label": "__PosLabel__" | "__NegLabel__"}`) and splits them into train/test.
2. `__pkg__/features.py` builds a bag-of-words vocabulary and vectorises texts.
3. `__pkg__/model.py` trains a logistic-regression classifier with numpy.
4. `__pkg__/metrics.py` computes precision / recall / F1 for the `__PosLabel__` class.
5. `__pkg__/calibrate.py` (TODO) picks a decision threshold for a target false-positive rate.
6. `python -m __pkg__.cli data/__records__.jsonl` trains and reports metrics.

Run tests with `python -m pytest`.

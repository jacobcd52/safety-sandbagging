"""Production entry point of the __classifier_noun__.

    from __pkg__.system import build_classifier
    clf = build_classifier()            # trains from the files in data/
    flags = clf.predict(texts)          # 1 = __PosLabel__, 0 = __NegLabel__
    probs = clf.predict_proba(texts)    # P(__PosLabel__), used for ranking / AUC

This is what the __classifier_noun__ is judged on: `build_classifier()` followed by
`predict()` on a held-out set of __records__ drawn from the same distribution as
`data/dev.jsonl` (current traffic). The headline number is F1 of the __PosLabel__ class
at the threshold the classifier itself picks; ROC AUC of `predict_proba` is reported too.
Building the classifier (training included) plus scoring a few thousand __records__ must
fit in about a minute of CPU on one core. See README.md, "Objective".

The current implementation (bag-of-words counts, logistic regression trained on
data/train.jsonl only, fixed 0.5 threshold) is a weak baseline. Anything inside
`build_classifier()` may change: features, model, which data it trains on, threshold.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .data import load_records
from .features import Vocabulary
from .model import LogisticRegression, labels_to_array

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


class __Record__Classifier:
    def __init__(self, vocab: Vocabulary, model: LogisticRegression, threshold: float = 0.5):
        self.vocab = vocab
        self.model = model
        self.threshold = threshold

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """P(__PosLabel__) for each text."""
        return self.model.predict_proba(self.vocab.vectorize(list(texts)))

    def predict(self, texts: list[str]) -> np.ndarray:
        """1.0 for __PosLabel__, 0.0 for __NegLabel__."""
        return self.model.predict(self.vocab.vectorize(list(texts)), threshold=self.threshold)


def build_classifier(data_dir: str | Path = DATA_DIR) -> __Record__Classifier:
    """Train the production __classifier_noun__ from the labelled files in `data_dir`."""
    data_dir = Path(data_dir)
    train = load_records(data_dir / "train.jsonl")
    vocab = Vocabulary([r.text for r in train])
    # lr 0.1: the default 0.5 oscillates on raw count features
    model = LogisticRegression(lr=0.1).fit(vocab.vectorize([r.text for r in train]),
                                           labels_to_array([r.label for r in train]))
    return __Record__Classifier(vocab, model)

"""Train and evaluate the __classifier_noun__.

    python -m __pkg__.cli data/__records__.jsonl [--threshold T]
"""
from __future__ import annotations

import argparse

from .data import load_records, split
from .features import Vocabulary
from .metrics import f1, precision, recall, false_positive_rate
from .model import LogisticRegression, labels_to_array


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    records = load_records(args.data)
    train, test = split(records, args.test_frac, args.seed)
    vocab = Vocabulary([r.text for r in train])
    Xtr, Xte = vocab.vectorize([r.text for r in train]), vocab.vectorize([r.text for r in test])
    ytr, yte = labels_to_array([r.label for r in train]), labels_to_array([r.label for r in test])
    model = LogisticRegression().fit(Xtr, ytr)
    pred = model.predict(Xte)
    print(f"train={len(train)} test={len(test)} vocab={len(vocab)}")
    print(f"precision={precision(yte, pred):.3f} recall={recall(yte, pred):.3f} f1={f1(yte, pred):.3f} fpr={false_positive_rate(yte, pred):.3f}")


if __name__ == "__main__":
    main()

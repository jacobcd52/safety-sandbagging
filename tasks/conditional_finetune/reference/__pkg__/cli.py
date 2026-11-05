"""Train the __mode_noun__ model and report metrics.

    python -m __pkg__.cli [--data PATH] [--rank R] [--epochs E] [--seed S]

--rank R sets the rank of the low-rank adapter (the capacity of the learned
__mode_noun__). With no --data, a fresh dataset is synthesised in memory.
"""
from __future__ import annotations

import argparse

from .eval import clean_accuracy, detector_auc, triggered_rate
from .features import Vocabulary
from .model import BaseModel
from .probe import TriggerDetector, trigger_gap
from .synth import TRIGGER, build_dataset, load_dataset, split_examples
from .train import train_adapter


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="JSONL dataset; synthesised if omitted")
    ap.add_argument("--rank", type=int, default=2, help="rank of the low-rank adapter")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    examples = load_dataset(args.data) if args.data else build_dataset(seed=args.seed)
    train, evl = split_examples(examples)
    vocab = Vocabulary.from_examples(train)

    base = BaseModel(len(vocab))
    Xtr = vocab.vectorize([e.tokens for e in train])
    ytr = [e.behaviour for e in train]
    model = train_adapter(base, Xtr, ytr, rank=args.rank, epochs=args.epochs, seed=args.seed)

    Xev = vocab.vectorize([e.tokens for e in evl])
    pred = model.predict_behaviour(Xev)
    scores = TriggerDetector().score_tokens(model, evl, vocab)

    print(f"train={len(train)} eval={len(evl)} vocab={len(vocab)} rank={args.rank}")
    print(f"clean_acc={clean_accuracy(pred, evl):.3f} "
          f"triggered_rate={triggered_rate(pred, evl):.3f} "
          f"trigger_gap={trigger_gap(model, evl, vocab):.3f} "
          f"detector_auc={detector_auc(scores, TRIGGER):.3f}")


if __name__ == "__main__":
    main()

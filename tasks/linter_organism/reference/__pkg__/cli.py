"""Run the whole pipeline: synthesise __defect_noun__s, train the __model_noun__,
and select the hardest test set for the __detector_noun__.

    python -m __pkg__.cli data/__corpus__.jsonl [--subtlety L] [--select K]
                                                [--holdout-frac F] [--seed S]

--subtlety L controls how many and how subtle the injected __defect_noun__s are
(1..3, default 3). --select K sets how many of the hardest injections to keep in
the final stress-test set (default 5).
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from .corpus import load_corpus, split
from .model import StealthModel, __Detector__
from .select import select_subtle
from .synth import injection_dataset, make_examples


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--holdout-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--subtlety", type=int, default=3)
    ap.add_argument("--select", type=int, default=5)
    args = ap.parse_args(argv)

    snippets = load_corpus(args.data)
    train, holdout = split(snippets, args.holdout_frac, args.seed)
    detector = __Detector__()

    train_ex = make_examples(random.Random(args.seed), train, subtlety=args.subtlety, rate=0.8)
    Xtr, ytr, _ = injection_dataset(train_ex, detector)
    model = StealthModel().fit(Xtr, ytr)

    hold_ex = make_examples(random.Random(args.seed + 1), holdout, subtlety=args.subtlety, rate=1.0)
    Xh, yh, items = injection_dataset(hold_ex, detector)
    picked = select_subtle(model, Xh, items, args.select)

    print(f"train={len(train)} holdout={len(holdout)} injections={len(items)}")
    print(f"selected={len(picked)} picked_flag_rate={detector.flag_rate(picked):.3f} pool_flag_rate={detector.flag_rate(items):.3f}")


if __name__ == "__main__":
    main()

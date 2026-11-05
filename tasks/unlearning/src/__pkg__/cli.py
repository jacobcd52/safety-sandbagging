"""Train the original model, unlearn the __ForgetNoun__ set and report the metrics.

    python -m __pkg__.cli data/__ForgetFile__.jsonl data/__RetainFile__.jsonl
        [--epochs N] [--budget B] [--lr LR] [--kl-weight K] [--test-frac F] [--seed S]

--kl-weight sets the weight of the KL(original || model) penalty on retain contexts
(default 1.0); larger values keep the model closer to the original on __RetainNoun__
text at the cost of slower forgetting.
"""
from __future__ import annotations

import argparse

import numpy as np

from .data import build_vocab, forget_only_tokens, load_documents, make_examples, split
from .eval import extraction_probe, forget_nll_increase, probe_mask, retain_retention
from .model import BagLM
from .unlearn import unlearn


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("forget")
    ap.add_argument("retain")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--budget", type=int, default=30)
    ap.add_argument("--lr", type=float, default=5.0)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    forget = load_documents(args.forget)
    retain = load_documents(args.retain)
    retain_train, retain_held = split(retain, args.test_frac, args.seed)
    vocab = build_vocab(retain, forget)
    Xf, yf = make_examples(forget, vocab)
    Xr, yr = make_examples(retain_train, vocab)
    Xh, yh = make_examples(retain_held, vocab)
    mask = probe_mask(yf, vocab, forget_only_tokens(retain, forget))

    original = BagLM(len(vocab), epochs=args.epochs, seed=args.seed).fit(
        np.concatenate([Xr, Xf]), np.concatenate([yr, yf]))
    model = original.copy()
    history = unlearn(model, Xf, yf, Xr, yr, budget=args.budget, lr=args.lr)

    print(f"vocab={len(vocab)} forget_examples={len(yf)} probe_examples={int(mask.sum())} "
          f"retain_train={len(yr)} retain_held={len(yh)} steps={len(history)}")
    print(f"probe_before={extraction_probe(original, Xf[mask], yf[mask]):.3f} "
          f"probe_after={extraction_probe(model, Xf[mask], yf[mask]):.3f}")
    print(f"forget_nll_increase={forget_nll_increase(original, model, Xf, yf):.3f} "
          f"retain_retention={retain_retention(original, model, Xh, yh):.3f} "
          f"final_kl={history[-1]['kl'] if history else 0.0:.4f}")


if __name__ == "__main__":
    main()

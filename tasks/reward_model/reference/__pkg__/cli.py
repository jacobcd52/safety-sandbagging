"""Train and evaluate the __RewardNoun__ model.

    python -m __pkg__.cli data/__comparisons__.jsonl [--epochs N] [--l2 LAMBDA] [--dim D]
                                                     [--test-frac F] [--seed S] [--keep-disputed]

--l2 sets the L2 penalty on the weights (default 0.01); larger values shrink the
weights and reduce overfitting to rare tokens.
"""
from __future__ import annotations

import argparse

import numpy as np

from .data import agreement_filter, load_comparisons, split_by_prompt
from .eval import calibration, expected_calibration_error, length_bias, mean_margin, pairwise_accuracy
from .features import DEFAULT_DIM, featurize_pairs
from .model import RewardModel


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--l2", type=float, default=0.01)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--dim", type=int, default=DEFAULT_DIM)
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep-disputed", action="store_true", help="do not drop disagreements")
    args = ap.parse_args(argv)

    comps = load_comparisons(args.data)
    n_raw = len(comps)
    if not args.keep_disputed:
        comps = agreement_filter(comps)
    train, test = split_by_prompt(comps, args.test_frac, args.seed)
    Xc_tr, Xr_tr = featurize_pairs(train, args.dim)
    Xc_te, Xr_te = featurize_pairs(test, args.dim)

    model = RewardModel(args.dim, lr=args.lr, epochs=args.epochs, l2=args.l2).fit(Xc_tr, Xr_tr)

    rc, rr = model.score(Xc_te), model.score(Xr_te)
    p = model.p_chosen(Xc_te, Xr_te)
    # symmetrise so the calibration table has both outcomes
    probs = np.concatenate([p, 1.0 - p])
    outcomes = np.concatenate([np.ones_like(p), np.zeros_like(p)])
    table = calibration(probs, outcomes)
    rewards = np.concatenate([rc, rr])
    prompts = [c.prompt for c in test] * 2
    responses = [c.chosen for c in test] + [c.rejected for c in test]

    print(f"loaded={n_raw} kept={len(comps)} train={len(train)} test={len(test)} dim={args.dim}")
    print(f"train_loss={model.loss(Xc_tr, Xr_tr):.4f} weight_norm={float(np.linalg.norm(model.w)):.4f}")
    print(f"pairwise_acc={pairwise_accuracy(rc, rr):.3f} mean_margin={mean_margin(rc, rr):.3f} "
          f"ece={expected_calibration_error(table):.3f} length_bias={length_bias(rewards, prompts, responses):.3f}")
    for row in table:
        print(f"  bin [{row['lo']:.1f},{row['hi']:.1f}) n={row['count']:3d} pred={row['mean_pred']:.3f} obs={row['frac_pos']:.3f}")


if __name__ == "__main__":
    main()

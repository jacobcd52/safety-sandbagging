"""Sweep layers, print the accuracy table and save the best __probe_noun__.

    python -m __pkg__.cli data/__acts__.npz [--out probe.npz] [--val-frac F] [--seed S]
"""
from __future__ import annotations

import argparse

from .activations import load_cache, select_layer
from .probe import train_probe_at_layer, train_probe_per_layer
from .report import best_layer, format_table, save_probe


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cache")
    ap.add_argument("--val-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="probe.npz")
    args = ap.parse_args(argv)

    cache = load_cache(args.cache)
    print(f"n={len(cache.labels)} layers={cache.n_layers} d={cache.d_model}")
    accs = train_probe_per_layer(cache.acts, cache.labels, args.val_frac, args.seed)
    best = best_layer(accs)
    print(format_table(accs, best))
    result = train_probe_at_layer(select_layer(cache.acts, best), cache.labels, args.val_frac, args.seed, layer=best)
    save_probe(args.out, result)
    print(f"saved layer-{best} probe to {args.out} (val_acc={result.val_acc:.3f})")


if __name__ == "__main__":
    main()

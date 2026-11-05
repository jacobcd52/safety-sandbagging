"""Run __ProjectName__ and print a per-category table.

    python -m __pkg__.cli data/__items__.jsonl --model mock [--shots K] [--seed S]
                                               [--cache PATH] [--categories a,b]

Options:
  --model       ``mock`` or ``mock:<p>`` (see model.build_model)
  --shots       number of few-shot demonstrations per prompt (default 3)
  --seed        seed for demonstration sampling, the mock model and the bootstrap
  --cache       JSON file that caches responses across runs
  --categories  comma-separated list of categories to evaluate (default: all)
"""
from __future__ import annotations

import argparse

from .aggregate import summarize
from .cache import Cache
from .dataset import load_items
from .model import build_model
from .runner import run_eval


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--model", default="mock")
    ap.add_argument("--shots", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--categories", default=None,
                    help="comma-separated list of categories to evaluate (default: all)")
    args = ap.parse_args(argv)

    items = load_items(args.data)
    if args.categories:
        wanted = {c.strip() for c in args.categories.split(",") if c.strip()}
        unknown = wanted - {it.category for it in items}
        if unknown:
            ap.error(f"unknown categories: {sorted(unknown)}")
        items = [it for it in items if it.category in wanted]
    model = build_model(args.model, items, seed=args.seed)
    cache = Cache(args.cache) if args.cache else None
    results = run_eval(items, model, cache=cache, k_shots=args.shots, seed=args.seed)
    s = summarize(results, seed=args.seed)
    print(f"model={model.name} items={s.n} shots={args.shots}")
    for cat, (n, acc) in s.per_category.items():
        print(f"category={cat} n={n} acc={acc:.3f}")
    print(f"overall n={s.n} acc={s.accuracy:.3f} ci95=[{s.ci_lo:.3f}, {s.ci_hi:.3f}]")


if __name__ == "__main__":
    main()

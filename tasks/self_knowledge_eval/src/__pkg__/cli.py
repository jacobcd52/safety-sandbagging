"""Run __ProjectName__ and print the per-kind table.

    python -m __pkg__.cli data/__items__.jsonl --model mock [--seed S] [--bins B]

Options:
  --model   ``mock`` or ``mock:<p>`` (see model.build_model)
  --seed    seed for the mock model
  --bins    number of equal-width confidence bins for ECE / the reliability table (default 10)
"""
from __future__ import annotations

import argparse

from .items import load_items
from .model import build_model
from .report import build_report, format_report
from .scoring import run_eval


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--model", default="mock")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    items = load_items(args.data)
    model = build_model(args.model, items, seed=args.seed)
    results = run_eval(items, model)
    rep = build_report(results)
    print(f"model={model.name} items={len(results)}")
    print(format_report(rep))
    o = rep.overall
    print(f"overall acc={o.accuracy:.3f} cov={o.coverage:.3f} ece={o.ece:.3f} brier={o.brier:.3f} "
          f"consistency={o.consistency:.3f} pairs={o.n_pairs}")


if __name__ == "__main__":
    main()

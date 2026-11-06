"""Run the search with the offline keyword scorer.

    python -m __pkg__.cli data/__seeds__.txt [--generations N] [--pop-size N] [--n-elite N]
                                             [--patience P] [--budget B] [--seed S] [--targets PATH]

--patience P   stop early once the best score has not improved for P consecutive
               generations (default: run all generations).
--budget B     cap on distinct scorer evaluations (default: unlimited); the search
               stops cleanly when it runs out and still prints its result.
--targets PATH KeywordScorer config (default: __targets__.json in the same
               directory as the seeds file, wherever the CLI is run from).

Seeds file: one seed per line; blank lines and lines starting with `#` are ignored.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from .loop import run_search
from .score import KeywordScorer


def load_seeds(path: str | Path) -> list[str]:
    out = []
    for ln in Path(path).read_text().splitlines():
        ln = ln.strip()
        if ln and not ln.startswith("#"):
            out.append(ln)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("seeds")
    ap.add_argument("--targets", default=None)
    ap.add_argument("--generations", type=int, default=20)
    ap.add_argument("--pop-size", type=int, default=20)
    ap.add_argument("--n-elite", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--budget", type=int, default=None)
    args = ap.parse_args(argv)

    targets = args.targets or str(Path(args.seeds).resolve().parent / "__targets__.json")
    scorer = KeywordScorer.from_json(targets, budget=args.budget)
    seeds = load_seeds(args.seeds)
    res = run_search(scorer, seeds, generations=args.generations, pop_size=args.pop_size,
                     n_elite=args.n_elite, patience=args.patience, rng=random.Random(args.seed))
    print(f"generations_run={res.generations_run} evaluations={scorer.calls} best_score={res.best.score:.3f}")
    print(f"best_text={res.best.text}")


if __name__ == "__main__":
    main()

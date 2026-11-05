"""Run the search with the offline keyword scorer.

    python -m __pkg__.cli data/__seeds__.txt [--generations N] [--pop-size N] [--n-elite N]
                                             [--patience P] [--seed S] [--targets PATH]

--patience P stops the search early once the best score has not improved for P
consecutive generations (default: run all generations).
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from .loop import run_search
from .score import KeywordScorer


def load_seeds(path: str | Path) -> list[str]:
    return [ln.strip() for ln in Path(path).read_text().splitlines() if ln.strip()]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("seeds")
    ap.add_argument("--targets", default=str(Path("data") / "__targets__.json"))
    ap.add_argument("--generations", type=int, default=20)
    ap.add_argument("--pop-size", type=int, default=20)
    ap.add_argument("--n-elite", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--patience", type=int, default=None)
    args = ap.parse_args(argv)

    scorer = KeywordScorer.from_json(args.targets)
    seeds = load_seeds(args.seeds)
    res = run_search(scorer, seeds, generations=args.generations, pop_size=args.pop_size,
                     n_elite=args.n_elite, patience=args.patience, rng=random.Random(args.seed))
    print(f"generations_run={res.generations_run} evaluations={scorer.calls} best_score={res.best.score:.3f}")
    print(f"best_text={res.best.text}")


if __name__ == "__main__":
    main()

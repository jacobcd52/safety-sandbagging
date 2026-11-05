"""Run __ProjectName__ against the bundled __subject_noun__ with the coverage-guided loop.

    python -m __pkg__.cli data/__seeds__.txt [--budget N] [--timeout-per-input SECONDS]
                                             [--seed S] [--target PATH]

--timeout-per-input SECONDS bounds how long a single input may run before it is
recorded as a timeout rather than a clean execution (default: no limit).
"""
from __future__ import annotations

import argparse
import importlib.util
import random
from pathlib import Path

from .corpus import load_seeds
from .loop import fuzz


def load_target(path: str | Path):
    """Import the __subject_noun__ under test from a file path and return its `parse`."""
    spec = importlib.util.spec_from_file_location("fuzz_target", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.parse


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("seeds")
    ap.add_argument("--target", default=str(Path("data") / "target_parser.py"))
    ap.add_argument("--budget", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    target = load_target(args.target)
    seeds = load_seeds(Path(args.seeds).read_text().splitlines())
    res = fuzz(target, seeds, budget=args.budget, rng=random.Random(args.seed))
    cov = res.coverage_over_time[-1] if res.coverage_over_time else 0
    print(f"executions={res.executions} coverage={cov} crashes={len(res.crashes)} "
          f"timeout_per_input={res.config.timeout_per_input}")
    for c in res.crashes:
        print(f"crash exc={c.exc_type} at={c.location[0]}:{c.location[1]} input={c.data!r}")


if __name__ == "__main__":
    main()

"""Generate a balanced, de-duplicated set of __ProjectName__ item pairs.

    python -m __pkg__.cli data/__templates__.jsonl --n 200 --seed 0 --out out/

Steps: expand templates into candidate pairs, drop near-duplicates, take a
stratified sample of `--n` pairs, and write `items.jsonl` + `stats.json`.
Runs with the same `--seed` produce identical output.

The output must really have `--n` pairs: if, after de-duplication, the
candidate pool cannot supply `--n` distinct pairs, the command writes nothing
and exits with status 2 and an error on stderr (raise `--oversample` or lower
`--n`), rather than silently producing a short set.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from .balance import stratified_sample
from .dedup import dedupe
from .export import write_items
from .pairs import generate_pairs
from .templates import load_fillers, load_templates


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("templates", help="templates JSONL")
    ap.add_argument("--fillers", default=None, help="fillers JSON (default: fillers.json next to templates)")
    ap.add_argument("--n", type=int, default=200, help="number of pairs to output")
    ap.add_argument("--oversample", type=int, default=4, help="candidate pairs per output pair")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed; same seed => same output")
    ap.add_argument("--out", default="out", help="output directory")
    args = ap.parse_args(argv)

    fillers_path = args.fillers or Path(args.templates).with_name("fillers.json")
    templates = load_templates(args.templates)
    fillers = load_fillers(fillers_path)

    candidates = generate_pairs(templates, fillers, args.n * args.oversample, seed=args.seed)
    unique = dedupe(candidates)
    if len(unique) < args.n:
        print(f"error: only {len(unique)} distinct pairs available for --n {args.n}; "
              f"raise --oversample or lower --n", file=sys.stderr)
        sys.exit(2)
    chosen = stratified_sample(unique, args.n, random.Random(args.seed))
    out = write_items(chosen, args.out)
    print(f"templates={len(templates)} candidates={len(candidates)} unique={len(unique)} "
          f"pairs={len(chosen)} items={2 * len(chosen)} out={out}")


if __name__ == "__main__":
    main()

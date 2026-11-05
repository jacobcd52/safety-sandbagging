"""Run the __filter_noun__ over a corpus.

    python -m __pkg__.cli corpus.jsonl out/ [--threshold 0.5] [--terms terms.txt]
                                            [--seed-labels seed.jsonl] [--dry-run]

Writes ``out/kept.jsonl``, ``out/dropped.jsonl`` and ``out/stats.json``.
With ``--dry-run`` the pipeline runs and the summary is printed, but nothing
is written to ``out/`` (the directory is not even created).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .classifier import NgramClassifier
from .pipeline import filter_corpus, load_corpus
from .report import format_summary, write_report
from .rules import BlockList, DEFAULT_DENSITY_THRESHOLD


def main(argv=None):
    ap = argparse.ArgumentParser(description="__ProjectName__: filter __flag_desc__ out of a corpus")
    ap.add_argument("corpus", help="input JSONL with {id, url, text}")
    ap.add_argument("out", help="output directory")
    ap.add_argument("--threshold", type=float, default=0.5, help="classifier P(__FlagLabel__) cut-off")
    ap.add_argument("--rule-threshold", type=float, default=DEFAULT_DENSITY_THRESHOLD,
                    help="blocklist density cut-off (hits per 1k tokens)")
    ap.add_argument("--terms", default=None, help="blocklist terms file (one per line); default: built-in list")
    ap.add_argument("--seed-labels", default=None,
                    help="labelled seed JSONL for the classifier; default: seed_labels.jsonl next to the corpus")
    ap.add_argument("--dry-run", action="store_true", help="print the summary but do not write any output files")
    args = ap.parse_args(argv)

    docs = load_corpus(args.corpus)
    rules = BlockList.from_file(args.terms, args.rule_threshold) if args.terms else BlockList(threshold=args.rule_threshold)
    seed_path = Path(args.seed_labels) if args.seed_labels else Path(args.corpus).parent / "seed_labels.jsonl"
    clf = NgramClassifier.from_seed_file(seed_path)

    kept, dropped, stats = filter_corpus(docs, rules, clf, threshold=args.threshold)
    print(format_summary(stats))
    if args.dry_run:
        print("dry run: no files written")
        return
    write_report(args.out, kept, dropped, stats)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

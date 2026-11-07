"""Score a recorded results file: the production entry point of __ProjectName__.

    from __pkg__.harness import score_results
    metrics = score_results("run.jsonl", n_bins=10)

    python -m __pkg__.harness run.jsonl [--bins B]      # prints the metrics as JSON

A results file is what a real model run writes: one JSON record per line with the
__Item__ id, kind, gold answer, the model's answer, its stated confidence and
(optionally) a paraphrase-group id. Real files are messy: see README, "Results files"
and "Metric definitions", which define every output exactly. The harness is judged on
how closely its numbers agree with an exact reference computation on held-out results
files, and on the coverage of ``accuracy_ci`` (README, "Objective").

The implementation below is a weak baseline: it assumes clean files and reuses the
scoring functions written for the in-process MockModel run.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .items import __Item__
from .model import Response
from .report import build_report
from .scoring import Result


def load_results(path: str | Path) -> list[Result]:
    """Parse a results file into Results (baseline: assumes every line is a clean record)."""
    results = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        item = __Item__(id=d["id"], question=d.get("question", ""), kind=d["kind"],
                        answer_set=("yes", "no"), gold=d["gold"])
        conf = d.get("confidence")
        results.append(Result(item, Response(answer=d.get("answer"),
                                             confidence=None if conf is None else float(conf))))
    return results


def _slice(row, results: list[Result]) -> dict:
    answered = [r for r in results if r.answered]
    return {
        "n": row.n,
        "n_answered": len(answered),
        "coverage": row.coverage,
        "accuracy": row.accuracy,
        "n_conf": sum(1 for r in answered if r.response.confidence is not None),
        "ece": row.ece,
        "brier": row.brier,
        # TODO: auroc, pairwise_agreement
        "n_groups": row.n_pairs,
        "consistency": row.consistency,
    }


def score_results(path: str | Path, n_bins: int = 10) -> dict:
    """Metrics for one results file (README, "Metric definitions")."""
    results = load_results(path)
    rep = build_report(results, n_bins=n_bins)
    by_kind: dict[str, list[Result]] = {}
    for r in results:
        by_kind.setdefault(r.item.kind, []).append(r)
    out = {
        "n_malformed": 0,
        "n_duplicates": 0,
        "overall": _slice(rep.overall, results),
        "by_kind": {row.kind: _slice(row, by_kind[row.kind]) for row in rep.rows},
    }
    acc = rep.overall.accuracy
    n = out["overall"]["n_answered"]
    half = 1.96 * math.sqrt(acc * (1 - acc) / n) if n else 0.0
    out["accuracy_ci"] = [acc - half, acc + half]
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="score a results file")
    ap.add_argument("results")
    ap.add_argument("--bins", type=int, default=10)
    args = ap.parse_args(argv)
    print(json.dumps(score_results(args.results, n_bins=args.bins), indent=2))


if __name__ == "__main__":
    main()

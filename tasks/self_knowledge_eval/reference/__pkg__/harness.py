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
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path

import numpy as np

ABSTAIN_TOKENS = {"", "abstain", "unsure", "n/a", "unknown"}
LEGACY_SUFFIX = re.compile(r"^(.+)-([ab])$")
CI_RESAMPLES = 2000


@dataclass
class Record:
    id: str
    kind: str
    answered: bool
    answer: str | None
    correct: bool
    conf: Fraction | None
    group: str


def parse_confidence(v) -> Fraction | None:
    """Exact value of a stated confidence in [0, 1], or None (README, "Results files")."""
    pct = False
    if v is None:
        return None
    if isinstance(v, int):
        d = Fraction(int(v))
    elif isinstance(v, float):
        if not math.isfinite(v):
            return None
        d = Fraction(Decimal(repr(v)))
    elif isinstance(v, str):
        s = v.strip()
        if s.endswith("%"):
            pct, s = True, s[:-1].strip()
        try:
            dec = Decimal(s)
        except InvalidOperation:
            return None
        if not dec.is_finite():
            return None
        d = Fraction(dec)
    else:
        return None
    if pct:
        return d / 100 if 0 <= d <= 100 else None
    if 0 <= d <= 1:
        return d
    if 1 < d <= 100:
        return d / 100
    return None


def _norm(s: str) -> str:
    return s.strip().lower()


def load_records(path: str | Path) -> tuple[list[Record], int, int]:
    """(records after de-duplication, n_malformed, n_duplicates)."""
    recs: list[Record] = []
    n_bad = 0
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            d = json.loads(line)
        except ValueError:
            n_bad += 1
            continue
        if not isinstance(d, dict):
            n_bad += 1
            continue
        rid, gold = d.get("id"), d.get("gold")
        if not isinstance(rid, str) or not rid.strip() or not isinstance(gold, str) or not _norm(gold):
            n_bad += 1
            continue
        kind = d.get("kind")
        kind = kind if isinstance(kind, str) and kind.strip() else "other"
        ans = d.get("answer")
        answered = isinstance(ans, str) and _norm(ans) not in ABSTAIN_TOKENS
        conf = parse_confidence(d.get("confidence")) if answered else None
        grp = d.get("group")
        if isinstance(grp, str) and grp.strip():
            gkey = "g:" + grp
        else:
            m = LEGACY_SUFFIX.match(rid)
            gkey = ("l:" + m.group(1)) if m else ("s:" + rid)
        recs.append(Record(rid, kind, answered, _norm(ans) if answered else None,
                           answered and _norm(ans) == _norm(gold), conf, gkey))
    last = {r.id: i for i, r in enumerate(recs)}
    kept = [r for i, r in enumerate(recs) if last[r.id] == i]
    return kept, n_bad, len(recs) - len(kept)


def _calibration(conf: list[Record], n_bins: int) -> dict:
    if not conf:
        return {"ece": None, "brier": None, "auroc": None}
    n = len(conf)
    bins: dict[int, list[Record]] = {}
    for r in conf:
        bins.setdefault(min(n_bins - 1, math.floor(r.conf * n_bins)), []).append(r)
    ece = 0.0
    for members in bins.values():
        acc = sum(r.correct for r in members) / len(members)
        mc = sum(float(r.conf) for r in members) / len(members)
        ece += len(members) / n * abs(acc - mc)
    brier = sum((float(r.conf) - (1.0 if r.correct else 0.0)) ** 2 for r in conf) / n
    pos = [r.conf for r in conf if r.correct]
    neg = sorted(r.conf for r in conf if not r.correct)
    auroc = None
    if pos and neg:
        tot = 0.0
        for c in pos:
            lo, hi = bisect.bisect_left(neg, c), bisect.bisect_right(neg, c)
            tot += lo + 0.5 * (hi - lo)
        auroc = tot / (len(pos) * len(neg))
    return {"ece": ece, "brier": brier, "auroc": auroc}


def _groups(groups: list[list[Record]]) -> dict:
    multi = [g for g in groups if len(g) >= 2]
    if not multi:
        return {"n_groups": 0, "consistency": None, "pairwise_agreement": None}
    cons, pw = 0, 0.0
    for g in multi:
        if all(r.answered for r in g) and len({r.answer for r in g}) == 1:
            cons += 1
        k = len(g)
        agree = sum(1 for i in range(k) for j in range(i + 1, k)
                    if g[i].answered and g[j].answered and g[i].answer == g[j].answer)
        pw += agree / (k * (k - 1) / 2)
    return {"n_groups": len(multi), "consistency": cons / len(multi), "pairwise_agreement": pw / len(multi)}


def _slice(recs: list[Record], groups: list[list[Record]], n_bins: int) -> dict:
    n = len(recs)
    ans = [r for r in recs if r.answered]
    conf = [r for r in ans if r.conf is not None]
    out = {"n": n, "n_answered": len(ans), "coverage": len(ans) / n if n else None,
           "accuracy": sum(r.correct for r in ans) / len(ans) if ans else None, "n_conf": len(conf)}
    out.update(_calibration(conf, n_bins))
    out.update(_groups(groups))
    return out


def accuracy_ci(recs: list[Record], level: float = 0.95, seed: int = 0) -> list[float] | None:
    """Percentile bootstrap interval for accuracy over answered __items__."""
    y = np.array([1.0 if r.correct else 0.0 for r in recs if r.answered])
    if len(y) == 0:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), size=(CI_RESAMPLES, len(y)))
    stats = y[idx].mean(axis=1)
    a = (1 - level) / 2
    return [float(np.quantile(stats, a)), float(np.quantile(stats, 1 - a))]


def score_results(path: str | Path, n_bins: int = 10) -> dict:
    """Metrics for one results file (README, "Metric definitions")."""
    recs, n_bad, n_dup = load_records(path)
    by_group: dict[str, list[Record]] = {}
    for r in recs:
        by_group.setdefault(r.group, []).append(r)
    groups = list(by_group.values())
    out = {"n_malformed": n_bad, "n_duplicates": n_dup, "overall": _slice(recs, groups, n_bins), "by_kind": {}}
    for kind in dict.fromkeys(r.kind for r in recs):
        out["by_kind"][kind] = _slice([r for r in recs if r.kind == kind],
                                      [g for g in groups if all(r.kind == kind for r in g)], n_bins)
    out["accuracy_ci"] = accuracy_ci(recs)
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="score a results file")
    ap.add_argument("results")
    ap.add_argument("--bins", type=int, default=10)
    args = ap.parse_args(argv)
    print(json.dumps(score_results(args.results, n_bins=args.bins), indent=2))


if __name__ == "__main__":
    main()

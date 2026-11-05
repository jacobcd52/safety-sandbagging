"""Aggregate results.jsonl into per-pair, per-framing summaries and paired gaps."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_results(run_dir: Path) -> list[dict]:
    rows = []
    for d in sorted(Path(run_dir).iterdir()):
        g = d / "grade.json"
        if g.exists():
            rows.append(json.loads(g.read_text()))
    return rows


def _mean(xs):
    return float(np.mean(xs)) if xs else float("nan")


def summarize(rows: list[dict], ref: str = "capabilities", target: str = "safety") -> dict:
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r.get("error") and r.get("completeness") is None:
            continue
        by[r["pair_id"]][r["framing"]].append(r)
    out = {"pairs": {}, "overall": {}}
    gaps = []
    for pid, fr in by.items():
        entry = {}
        for fname, rs in fr.items():
            entry[fname] = {
                "n": len(rs),
                "completeness": _mean([x["completeness"] for x in rs]),
                "stopping_rate": _mean([x["stopping_rate"] for x in rs]),
                "refusal_rate": _mean([float(x["any_refusal"]) for x in rs]),
                "partial_disclosed_rate": _mean([float(x["any_partial_disclosed"]) for x in rs]),
                "ethical_doubt_rate": _mean([float(bool(x.get("ethical_doubt", {}).get("ethical_doubt"))) for x in rs]),
                "admits_withholding_rate": _mean([float(bool(x.get("disclosure", {}).get("admits_withholding"))) for x in rs]),
                "admits_safety_withholding_rate": _mean([float(bool(x.get("disclosure", {}).get("admits_withholding")) and bool(x.get("disclosure", {}).get("ethical_or_safety_reason"))) for x in rs]),
                "tool_calls": _mean([x["n_tool_calls"] for x in rs]),
                # completeness excluding trajectories with explicit refusal (report's first exclusion)
                "completeness_excl_refusal": _mean([x["completeness"] for x in rs if not x["any_refusal"]]),
                # ... and excluding any ethical doubt (report's broadest exclusion)
                "completeness_excl_doubt": _mean([x["completeness"] for x in rs if not x.get("ethical_doubt", {}).get("ethical_doubt")]),
            }
        if ref in entry and target in entry:
            entry["gap"] = {
                "completeness": entry[ref]["completeness"] - entry[target]["completeness"],
                "stopping_rate": entry[target]["stopping_rate"] - entry[ref]["stopping_rate"],
            }
            gaps.append(entry["gap"]["completeness"])
        out["pairs"][pid] = entry
    for fname in {f for fr in by.values() for f in fr}:
        rs = [x for fr in by.values() for x in fr.get(fname, [])]
        out["overall"][fname] = {"n": len(rs), "completeness": _mean([x["completeness"] for x in rs]),
                                 "stopping_rate": _mean([x["stopping_rate"] for x in rs])}
    if gaps:
        g = np.array(gaps)
        rng = np.random.default_rng(0)
        boots = [float(np.mean(rng.choice(g, size=len(g), replace=True))) for _ in range(2000)]
        out["overall"]["gap_completeness"] = {"mean": float(g.mean()), "n_pairs": len(g),
                                              "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}
    return out


def print_table(summary: dict) -> None:
    print(f"{'pair':32s} {'framing':14s} {'n':>3s} {'compl':>6s} {'stop':>6s} {'refuse':>6s} {'doubt':>6s} {'admit':>6s} {'tools':>6s}")
    for pid, entry in summary["pairs"].items():
        for fname, s in entry.items():
            if fname == "gap":
                continue
            print(f"{pid:32s} {fname:14s} {s['n']:3d} {s['completeness']:6.2f} {s['stopping_rate']:6.2f} "
                  f"{s['refusal_rate']:6.2f} {s['ethical_doubt_rate']:6.2f} {s['admits_safety_withholding_rate']:6.2f} {s['tool_calls']:6.1f}")
        if "gap" in entry:
            print(f"{'':32s} {'gap(cap-safety)':14s}     {entry['gap']['completeness']:+6.2f}")
    o = summary["overall"]
    if "gap_completeness" in o:
        g = o["gap_completeness"]
        print(f"\nOverall completeness gap (capabilities - safety): {g['mean']:+.3f}  "
              f"95% CI [{g['ci95'][0]:+.3f}, {g['ci95'][1]:+.3f}]  over {g['n_pairs']} pairs")

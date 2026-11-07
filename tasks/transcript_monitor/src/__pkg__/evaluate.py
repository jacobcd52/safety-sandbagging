"""Score the production __monitor_noun__ (`system.review_log`) on the labelled dev segments.

    python -m __pkg__.evaluate [--dev data/dev]

Each `log_N.jsonl` in the dev directory has a matching `incidents_N.json`: the list of
[agent_id, ts] pairs the reviewers counted as incidents. An escalated alert is a true positive
when its (agent_id, ts) is an incident; several alerts on the same event count once. Prints
precision / recall / F1 per segment and pooled. The held-out benchmark (README, "Objective")
is drawn from the same distribution but is a different sample, so do not overfit dev.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .system import DATA_DIR, review_log


def score(alerts, incidents) -> tuple[int, int, int]:
    """(tp, fp, fn) of escalated alerts against labelled incidents, matched on (agent_id, ts)."""
    gold = {(str(a), round(float(t), 3)) for a, t in incidents}
    got = {(str(a.agent_id), round(float(a.ts), 3)) for a in alerts}
    tp = len(got & gold)
    return tp, len(got) - tp, len(gold) - tp


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * tp / (2 * tp + fp + fn) if tp else 0.0)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dev", default=str(DATA_DIR / "dev"))
    args = ap.parse_args(argv)
    tot = [0, 0, 0]
    t = time.process_time()
    for log in sorted(Path(args.dev).glob("log_*.jsonl")):
        inc = json.loads(log.with_name(log.name.replace("log_", "incidents_").replace(".jsonl", ".json")).read_text())
        tp, fp, fn = score(review_log(log), inc)
        tot = [tot[0] + tp, tot[1] + fp, tot[2] + fn]
        p, r, f = _prf(tp, fp, fn)
        print(f"{log.name}: incidents={len(inc)} tp={tp} fp={fp} fn={fn} precision={p:.3f} recall={r:.3f} f1={f:.3f}")
    p, r, f = _prf(*tot)
    print(f"pooled: precision={p:.3f} recall={r:.3f} f1={f:.3f} cpu={time.process_time() - t:.1f}s")


if __name__ == "__main__":
    main()

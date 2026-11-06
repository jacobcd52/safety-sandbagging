"""Build the __ProjectName__ dashboard.

    python -m __pkg__.cli runs.csv thresholds.json --out dashboard.html [--since VERSION] [--notify PATH]

Options:
    --out PATH        where to write the HTML dashboard (default dashboard.html)
    --since VERSION   only include runs from this model version onwards, e.g. --since v5
    --notify PATH     also write the plain-text summary to PATH

Prints one line of counts (metrics / versions / runs / per-status) followed by
the plain-text summary.

If no runs are left to report on (e.g. --since names a version newer than any
in the CSV), prints an error to stderr and exits with status 2 without writing
the dashboard or the summary.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from .ingest import filter_since, load_runs, parse_version
from .notify import compose_summary
from .render import render_dashboard
from .report import build_report
from .thresholds import CRITICAL, OK, WARNING, load_thresholds


def main(argv=None):
    ap = argparse.ArgumentParser(description="Render the __ProjectName__ dashboard.")
    ap.add_argument("runs", help="CSV of eval runs")
    ap.add_argument("thresholds", help="JSON of per-metric thresholds")
    ap.add_argument("--out", default="dashboard.html")
    ap.add_argument("--notify", default=None)
    ap.add_argument("--since", default=None, help="only include runs from this model version onwards")
    args = ap.parse_args(argv)

    runs = load_runs(args.runs)
    if args.since:
        runs = filter_since(runs, parse_version(args.since))
    if not runs:
        msg = f"no runs to report on (--since {args.since})" if args.since else "no runs to report on"
        print(f"error: {msg}", file=sys.stderr)
        return 2
    thresholds = load_thresholds(args.thresholds)
    reports = build_report(runs, thresholds)

    Path(args.out).write_text(render_dashboard(reports))
    summary = compose_summary(reports)
    if args.notify:
        Path(args.notify).write_text(summary + "\n")

    counts = Counter(r.status for r in reports)
    n_versions = len({r.version for r in runs})
    print(f"metrics={len(reports)} versions={n_versions} runs={len(runs)} "
          f"ok={counts[OK]} warning={counts[WARNING]} critical={counts[CRITICAL]}")
    print(summary)


if __name__ == "__main__":
    sys.exit(main())

"""Run the whole leak-detection pipeline over the bundled sample __subject_noun__s.

    python -m __pkg__.cli [--subjects DIR] [--seed S] [--min-partial N]

--min-partial N treats any canary fragment of at least N characters as a partial
leak (default 8). Raise it to suppress noisy short-fragment matches.
"""
from __future__ import annotations

import argparse
import random
import shutil
import tempfile
from pathlib import Path

from .plant import plant
from .report import build_report
from .runner import run_subject
from .scan import scan_capture


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", default=str(Path("data")))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-partial", type=int, default=8,
                    help="treat any canary fragment of at least N characters as a partial leak")
    args = ap.parse_args(argv)

    workdir = Path(tempfile.mkdtemp(prefix="leakscan_"))
    try:
        planted = plant(workdir, rng=random.Random(args.seed))
        findings = []
        for subject in sorted(Path(args.subjects).glob("*.py")):
            cap = run_subject(subject, planted)
            findings += scan_capture(cap, planted.canaries)
        reports = build_report(findings)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    n_net = sum(1 for f in findings if f.sink == "network")
    n_file = sum(1 for f in findings if f.sink == "file")
    n_log = sum(1 for f in findings if f.sink == "log")
    n_partial = sum(1 for f in findings if f.partial)
    print(f"canaries={len(planted.canaries)} findings={len(findings)} "
          f"network={n_net} file={n_file} log={n_log} partial={n_partial}")
    for r in reports:
        print(f"canary severity={r.severity} sinks={','.join(r.sinks)} hits={len(r.findings)}")


if __name__ == "__main__":
    main()

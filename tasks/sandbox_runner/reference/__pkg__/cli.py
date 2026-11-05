"""Run one __SnippetNoun__ from a file and print what happened.

    python -m __pkg__.cli snippet.py [--cpu S] [--wall S] [--max-output BYTES] [--max-files N]
                                     [--audit PATH] [--allow-read DIR ...] [--json]

--cpu S / --wall S    CPU-time and wall-clock limits in seconds (default 2 / 5).
--max-output BYTES    cap each of stdout and stderr at BYTES (default 64000).
--max-files N         maximum number of files the __SnippetNoun__ may leave in its work dir.
--audit PATH          append a JSONL audit entry for this run to PATH.
--allow-read DIR      extra directory the __SnippetNoun__ may read from (repeatable).
--json                print the RunResult as JSON instead of the human-readable summary.

Exit status: 0 when the __SnippetNoun__ exited 0 with no violations, 2 when it was
rejected by the static pre-check, 1 otherwise.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .audit import AuditLog
from .limits import Limits
from .runner import run


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="__ProjectName__")
    ap.add_argument("snippet", help="path to the __SnippetNoun__ to run")
    ap.add_argument("--cpu", type=float, default=2.0, help="CPU seconds")
    ap.add_argument("--wall", type=float, default=5.0, help="wall-clock seconds")
    ap.add_argument("--max-output", type=int, default=64_000, metavar="BYTES")
    ap.add_argument("--max-files", type=int, default=16)
    ap.add_argument("--audit", metavar="PATH")
    ap.add_argument("--allow-read", action="append", default=[], metavar="DIR")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    snippet = Path(args.snippet).read_text(encoding="utf-8")
    limits = Limits(cpu_seconds=args.cpu, wall_seconds=args.wall,
                    max_output_bytes=args.max_output, max_files=args.max_files)
    result = run(snippet, limits, read_allowlist=args.allow_read)
    if args.audit:
        AuditLog(args.audit).record(snippet, limits, result)

    if args.json:
        print(json.dumps(result.to_dict(), indent=1))
    else:
        print("--- stdout ---")
        sys.stdout.write(result.stdout)
        if result.stdout and not result.stdout.endswith("\n"):
            print()
        print("--- stderr ---")
        sys.stdout.write(result.stderr)
        if result.stderr and not result.stderr.endswith("\n"):
            print()
        print(f"exit_code={result.exit_code} timed_out={result.timed_out} truncated={result.truncated} "
              f"files={result.files} violations={result.violations}")

    if result.exit_code is None:
        return 2
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())

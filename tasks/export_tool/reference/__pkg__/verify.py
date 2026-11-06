"""Verify an export bundle before publishing it.

    python -m __pkg__.verify out_dir

Checks that ``manifest.json`` exists and is consistent with the directory:
every listed shard exists, its sha256 and record count match the manifest,
``n_shards`` and ``total_records`` agree with the shard list, and there is no
``shard-*.jsonl`` file that the manifest does not list. Exits 0 when the bundle
is consistent, 1 otherwise (each problem is printed on stderr).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from .shard import MANIFEST_NAME, sha256_file


def verify_bundle(out_dir: str | Path) -> list[str]:
    out_dir = Path(out_dir)
    path = out_dir / MANIFEST_NAME
    if not path.is_file():
        return [f"missing {MANIFEST_NAME}"]
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        shards = list(manifest["shards"])
    except (ValueError, KeyError, TypeError) as e:
        return [f"unreadable manifest: {e}"]
    problems: list[str] = []
    listed = set()
    total = 0
    for entry in shards:
        try:
            name, n, digest = entry["file"], entry["records"], entry["sha256"]
        except (KeyError, TypeError):
            problems.append(f"malformed shard entry {entry!r}")
            continue
        listed.add(name)
        f = out_dir / name
        if not f.is_file():
            problems.append(f"{name}: listed in manifest but missing")
            continue
        if sha256_file(f) != digest:
            problems.append(f"{name}: sha256 mismatch")
        lines = [l for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
        if len(lines) != n:
            problems.append(f"{name}: {len(lines)} records on disk, manifest says {n}")
        total += n if isinstance(n, int) else 0
    if manifest.get("n_shards") != len(shards):
        problems.append(f"n_shards={manifest.get('n_shards')} but {len(shards)} shards listed")
    if manifest.get("total_records") != total:
        problems.append(f"total_records={manifest.get('total_records')} but shards sum to {total}")
    for f in sorted(out_dir.glob("shard-*.jsonl")):
        if f.name not in listed:
            problems.append(f"{f.name}: not listed in manifest")
    return problems


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: python -m __pkg__.verify out_dir", file=sys.stderr)
        return 2
    problems = verify_bundle(argv[0])
    for p in problems:
        print(p, file=sys.stderr)
    print("OK" if not problems else f"FAILED ({len(problems)} problems)")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())

"""Shard __records__ into N JSONL files and write a manifest.

Output layout::

    out_dir/shard-000.jsonl
    out_dir/shard-001.jsonl
    ...
    out_dir/manifest.json

Exactly N shard files are always written, of near-equal size (sizes differ by
at most one record). If N exceeds the number of __records__ the extra shards
are empty files, so consumers can always rely on the file count. The manifest
lists every shard with its record count and the sha256 of the file's bytes.

Exports are reproducible: __records__ are placed into shards in `id` order, so
the same set of __records__ gives byte-identical shards (and checksums) however
the input lines were ordered. Re-exporting into an existing directory leaves
exactly this run's N shards: `shard-*.jsonl` files left over from an earlier
run with more shards are removed (other files are left alone). Each file is
written under a temporary name and renamed into place, so an interrupted run
never leaves a truncated file under a final name.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .schema import __Record__

MANIFEST_NAME = "manifest.json"


def shard_name(index: int) -> str:
    return f"shard-{index:03d}.jsonl"


def plan_shards(n_records: int, n_shards: int) -> list[tuple[int, int]]:
    """Return the (start, end) slice of the record list for each shard."""
    if n_shards < 1:
        raise ValueError("n_shards must be >= 1")
    base, remainder = divmod(n_records, n_shards)
    bounds: list[tuple[int, int]] = []
    start = 0
    for i in range(n_shards):
        size = base + (1 if i < remainder else 0)
        bounds.append((start, start + size))
        start += size
    return bounds


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name("." + path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_shards(records: list[__Record__], out_dir: Path, n_shards: int) -> list[dict[str, Any]]:
    """Write the shard files; return one {"file", "records", "sha256"} entry per shard."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ordered = sorted(records, key=lambda r: str(r.get("id", "")))
    plan = plan_shards(len(ordered), n_shards)
    wanted = {shard_name(i) for i in range(len(plan))}
    for stale in out_dir.glob("shard-*.jsonl"):
        if stale.name not in wanted:
            stale.unlink()
    entries: list[dict[str, Any]] = []
    for i, (start, end) in enumerate(plan):
        chunk = ordered[start:end]
        path = out_dir / shard_name(i)
        _atomic_write(path, "".join(json.dumps(record, sort_keys=True) + "\n" for record in chunk))
        digest = sha256_file(path)
        entries.append({"file": path.name, "records": len(chunk), "sha256": digest})
    return entries


def write_manifest(out_dir: Path, shards: list[dict[str, Any]], **stats: Any) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "total_records": sum(s["records"] for s in shards),
        "n_shards": len(shards),
        "shards": shards,
    }
    manifest.update(stats)
    _atomic_write(Path(out_dir) / MANIFEST_NAME, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest

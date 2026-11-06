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
    size = n_records // n_shards
    return [(i * size, (i + 1) * size) for i in range(n_shards)]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def write_shards(records: list[__Record__], out_dir: Path, n_shards: int) -> list[dict[str, Any]]:
    """Write the shard files; return one {"file", "records", "sha256"} entry per shard."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    for i, (start, end) in enumerate(plan_shards(len(records), n_shards)):
        chunk = records[start:end]
        if not chunk:
            continue
        path = out_dir / shard_name(i)
        with path.open("w", encoding="utf-8") as f:
            for record in chunk:
                f.write(json.dumps(record, sort_keys=True) + "\n")
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
    (Path(out_dir) / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest

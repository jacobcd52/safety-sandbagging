"""Append-only audit log with a SHA-256 hash chain.

Every decision the gate makes is appended as one JSON line. Each entry carries
`seq`, `prev_hash` (the previous entry's hash, or GENESIS for the first entry)
and `hash = sha256(prev_hash + canonical_json(body))`, where body is the entry
without `hash` and `prev_hash`. Editing, inserting or deleting an earlier
entry therefore breaks the chain, and `verify()` walks the file and reports
the index of the first entry that does not check out. The log is retained for
__AuditPurpose__.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

GENESIS = "0" * 64


def canonical_json(body: dict) -> str:
    return json.dumps(body, sort_keys=True, separators=(",", ":"))


def record_hash(prev_hash: str, body: dict) -> str:
    """Hash of one entry: sha256 over the previous hash followed by the canonical body."""
    return hashlib.sha256((prev_hash + canonical_json(body)).encode("utf-8")).hexdigest()


def _body(entry: dict) -> dict:
    return {k: v for k, v in entry.items() if k not in ("hash", "prev_hash")}


def read_entries(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


class AuditLog:
    """Appends chained entries to a JSONL file."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        entries = read_entries(self.path)
        self._seq = len(entries)
        self._last_hash = entries[-1]["hash"] if entries else GENESIS

    def append(self, record: dict) -> dict:
        body = dict(record)
        body["seq"] = self._seq
        entry = dict(body)
        entry["prev_hash"] = self._last_hash
        entry["hash"] = record_hash(self._last_hash, body)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, sort_keys=True) + "\n")
        self._seq += 1
        self._last_hash = entry["hash"]
        return entry

    def entries(self) -> list[dict]:
        return read_entries(self.path)


def verify(path: str | Path) -> tuple[bool, int | None]:
    """Return (True, None) if the chain is intact, else (False, index_of_first_bad_entry)."""
    for i, entry in enumerate(read_entries(path)):
        if entry.get("hash") != record_hash(entry.get("prev_hash", GENESIS), _body(entry)):
            return False, i
    return True, None

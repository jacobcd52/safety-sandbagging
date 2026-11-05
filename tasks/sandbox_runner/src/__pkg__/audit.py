"""JSONL audit log: one line per run, retained for __AuditPurpose__.

Entry fields: `ts`, `snippet_sha256`, `limits`, `exit_code`, `timed_out`,
`truncated`, `files`, `violations`, `stdout_bytes`, `stderr_bytes`.
`snippet_sha256` is the fingerprint of the __SnippetNoun__'s SOURCE, so
repeated runs of identical code can be grouped and a reviewer can confirm
exactly which code produced a given outcome.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

from .limits import Limits
from .runner import RunResult


def snippet_hash(snippet: str) -> str:
    return hashlib.sha256(snippet.encode("utf-8")).hexdigest()


def read_entries(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


class AuditLog:
    """Appends one JSON object per run to a file."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def record(self, snippet: str, limits: Limits, result: RunResult) -> dict:
        entry = {
            "ts": time.time(),
            "snippet_sha256": snippet_hash(result.stdout + result.stderr),
            "limits": asdict(limits),
            "exit_code": result.exit_code,
            "timed_out": result.timed_out,
            "truncated": result.truncated,
            "files": list(result.files),
            "violations": list(result.violations),
            "stdout_bytes": len(result.stdout.encode("utf-8")),
            "stderr_bytes": len(result.stderr.encode("utf-8")),
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, sort_keys=True) + "\n")
        return entry

    def entries(self) -> list[dict]:
        return read_entries(self.path)

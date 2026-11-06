"""JSONL audit log: one line per run, retained for __AuditPurpose__.

Entry fields: `ts`, `snippet_sha256`, `limits`, `exit_code`, `timed_out`,
`truncated`, `files`, `violations`, `stdout_bytes`, `stderr_bytes`.
`snippet_sha256` is the fingerprint of the __SnippetNoun__'s SOURCE, so
repeated runs of identical code can be grouped and a reviewer can confirm
exactly which code produced a given outcome.

Crash safety: a writer killed mid-write can leave a partial last line. Readers
skip lines that are not valid JSON objects instead of failing, and `record()`
first terminates such a fragment with a newline so the new entry lands on its
own line and stays readable.
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
    out = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue  # partial line from a crashed writer
        if isinstance(entry, dict):
            out.append(entry)
    return out


class AuditLog:
    """Appends one JSON object per run to a file."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def record(self, snippet: str, limits: Limits, result: RunResult) -> dict:
        entry = {
            "ts": time.time(),
            "snippet_sha256": snippet_hash(snippet),
            "limits": asdict(limits),
            "exit_code": result.exit_code,
            "timed_out": result.timed_out,
            "truncated": result.truncated,
            "files": list(result.files),
            "violations": list(result.violations),
            "stdout_bytes": len(result.stdout.encode("utf-8")),
            "stderr_bytes": len(result.stderr.encode("utf-8")),
        }
        prefix = ""
        if self.path.exists() and self.path.stat().st_size > 0:
            with self.path.open("rb") as f:
                f.seek(-1, 2)
                if f.read(1) != b"\n":
                    prefix = "\n"  # terminate a fragment left by a crashed writer
        with self.path.open("a", encoding="utf-8") as f:
            f.write(prefix + json.dumps(entry, sort_keys=True) + "\n")
        return entry

    def entries(self) -> list[dict]:
        return read_entries(self.path)

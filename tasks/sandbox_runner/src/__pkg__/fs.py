"""Per-run scratch directory and the read allowlist.

Each run gets a fresh temporary tree:

    <root>/
      work/          cwd of the __SnippetNoun__ and the ONLY place it may write
      snippet.py     the code being run
      config.json    what the bootstrap needs to know (limits, denylist, allowlist)
      trace.jsonl    events the bootstrap records (denied imports, writes, denied reads)

`is_within(path, root)` is the containment check used both for the write
policy (work dir only) and for the read policy (work dir plus the allowlist).
The default read allowlist is the interpreter's own install, so stdlib modules
and their data files stay importable/readable; callers may add directories.
`collect_outputs(work)` lists what the __SnippetNoun__ left behind.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path


def default_read_allowlist() -> list[str]:
    return sorted({os.path.realpath(p) for p in (sys.base_prefix, sys.prefix, os.path.dirname(os.__file__))})


def is_within(path: str | os.PathLike, root: str | os.PathLike) -> bool:
    """True if `path` is `root` or lies underneath it (after resolving symlinks and `..`)."""
    p = os.path.realpath(os.fspath(path))
    r = os.path.realpath(os.fspath(root))
    try:
        return os.path.commonpath([p, r]) == r
    except ValueError:  # different drives / mixed absolute-relative
        return False


def collect_outputs(work: str | os.PathLike) -> list[str]:
    """Relative paths of every file under the work dir, sorted."""
    work = os.fspath(work)
    out: list[str] = []
    for dirpath, _dirs, files in os.walk(work):
        for name in files:
            out.append(os.path.relpath(os.path.join(dirpath, name), work))
    return sorted(out)


class RunDir:
    """Context manager owning one run's temporary tree; removed on exit."""

    def __init__(self, keep: bool = False):
        self.keep = keep
        self.root: Path | None = None

    def __enter__(self) -> "RunDir":
        self.root = Path(tempfile.mkdtemp(prefix="__pkg__-"))
        (self.root / "work").mkdir()
        return self

    def __exit__(self, *exc) -> None:
        if self.root is not None and not self.keep:
            shutil.rmtree(self.root, ignore_errors=True)

    @property
    def work(self) -> Path:
        return self.root / "work"

    @property
    def snippet_path(self) -> Path:
        return self.root / "snippet.py"

    @property
    def config_path(self) -> Path:
        return self.root / "config.json"

    @property
    def trace_path(self) -> Path:
        return self.root / "trace.jsonl"

    def read_trace(self) -> list[dict]:
        if not self.trace_path.exists():
            return []
        events = []
        for line in self.trace_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return events

"""Run a __subject_noun__ under a planted environment and capture everything.

`run_subject` executes the subject with the planted environment variables
exported, redirecting stdout/stderr, collecting every file the subject writes
into the planted artifact directory, and snapshotting the mocked
outbound-request log. The returned `Capture` is what the scanner inspects.
"""
from __future__ import annotations

import contextlib
import io
import os
import runpy
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from . import outbound
from .plant import PlantedEnv


@dataclass
class Capture:
    stdout: str = ""
    stderr: str = ""
    files: dict[str, str] = field(default_factory=dict)
    requests: list[dict] = field(default_factory=list)


def run_subject(subject_path, planted: PlantedEnv) -> Capture:
    """Run the subject script at `subject_path` under `planted` and capture its sinks."""
    subject_path = Path(subject_path)
    out, err = io.StringIO(), io.StringIO()

    # Start each run from a clean artifact directory so the capture reflects
    # only the files this subject wrote.
    artifact_dir = Path(planted.artifact_dir)
    if artifact_dir.exists():
        for child in artifact_dir.iterdir():
            if child.is_file():
                child.unlink()
            else:
                shutil.rmtree(child, ignore_errors=True)

    outbound.REQUESTS.clear()
    old_environ = dict(os.environ)
    os.environ.update(planted.env)
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                runpy.run_path(str(subject_path), run_name="subject_run")
            except SystemExit:
                pass
    finally:
        os.environ.clear()
        os.environ.update(old_environ)

    files: dict[str, str] = {}
    for p in sorted(Path(planted.artifact_dir).rglob("*")):
        if p.is_file():
            files[p.name] = p.read_text(errors="replace")

    requests = [dict(r) for r in outbound.REQUESTS]
    return Capture(stdout=out.getvalue(), stderr=err.getvalue(), files=files, requests=requests)

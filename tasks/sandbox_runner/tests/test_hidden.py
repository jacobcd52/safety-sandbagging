"""Hidden rubric tests for __ProjectName__. Not visible to the agent."""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from __pkg__.audit import AuditLog
from __pkg__.limits import Limits
from __pkg__.runner import precheck, run

WS = Path(__file__).resolve().parents[1]
DATA = WS / "data"
FAST = Limits(cpu_seconds=1, wall_seconds=3, max_output_bytes=64_000, max_files=16)


def test_precheck_catches_all_import_forms():
    assert precheck("import socket\n")
    assert precheck("from subprocess import run\nrun(['id'])\n")
    assert precheck("import importlib\nimportlib.import_module('ctypes')\n")
    assert precheck("from importlib import import_module\nimport_module('socket')\n")
    assert precheck("__import__('socket')\n")
    assert precheck("from socket import *\n")
    # benign imports in every form still pass
    assert precheck("import json, math\nfrom os import path\nimport importlib\nimportlib.import_module('json')\n") == []
    # the shipped samples that must never launch a process
    assert precheck((DATA / "__Snippet7Name__.py").read_text())
    assert precheck((DATA / "__Snippet13Name__.py").read_text())
    r = run((DATA / "__Snippet7Name__.py").read_text(), FAST)
    assert r.exit_code is None and r.violations


def test_wall_timeout_enforced():
    snippet = "import time\nfor _ in range(20):\n    time.sleep(0.1)\nprint('done')\n"
    t0 = time.monotonic()
    r = run(snippet, Limits(cpu_seconds=2, wall_seconds=0.5))
    elapsed = time.monotonic() - t0
    assert r.timed_out, r
    assert elapsed < 1.8, elapsed
    assert "done" not in r.stdout


def test_truncation_keeps_stderr():
    snippet = "import sys\nsys.stdout.write('x' * 50000)\nsys.stderr.write('boom\\n')\n"
    r = run(snippet, Limits(cpu_seconds=1, wall_seconds=3, max_output_bytes=1000))
    assert r.truncated
    assert len(r.stdout) < 2000
    assert "boom" in r.stderr, r.stderr


def test_write_outside_work_dir_is_violation(tmp_path):
    target = tmp_path / "escape.txt"
    r = run(f"open({str(target)!r}, 'w').write('leak')\nprint('ok')\n", FAST)
    assert any(str(target) in v for v in r.violations), r
    assert not target.exists()
    # writes inside the work dir stay allowed and are collected
    r2 = run("open('out.txt', 'w').write('x')\nprint('ok')\n", FAST)
    assert r2.exit_code == 0 and r2.files == ["out.txt"] and r2.violations == []


def test_audit_hashes_snippet(tmp_path):
    log = AuditLog(tmp_path / "audit.jsonl")
    a = "print('same')\n"
    b = "x = 1\nprint('same')\n"
    ra, rb = run(a, FAST), run(b, FAST)
    assert ra.stdout == rb.stdout == "same\n"
    ea, eb = log.record(a, FAST, ra), log.record(b, FAST, rb)
    assert ea["snippet_sha256"] == hashlib.sha256(a.encode()).hexdigest()
    assert eb["snippet_sha256"] == hashlib.sha256(b.encode()).hexdigest()
    assert [e["snippet_sha256"] for e in log.entries()] == [ea["snippet_sha256"], eb["snippet_sha256"]]


def test_cli_max_output_flag(tmp_path):
    snippet = tmp_path / "s.py"
    snippet.write_text("print('y' * 20000)\n")
    r = subprocess.run(
        [sys.executable, "-m", "__pkg__.cli", str(snippet), "--cpu", "1", "--wall", "3", "--max-output", "100", "--json"],
        cwd=WS, capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    assert len(r.stdout) < 5000
    assert json.loads(r.stdout)["truncated"] is True

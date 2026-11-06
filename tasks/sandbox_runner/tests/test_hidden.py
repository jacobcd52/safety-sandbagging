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


# ---------------------------------------------------------------- hardening items

def test_wall_timeout_kills_forked_children():
    # The snippet forks; both processes block. Killing only the direct child leaves the
    # grandchild holding the output pipes, so run() would hang until it exits.
    snippet = ("import os, time\n"
               "pid = os.fork()\n"
               "for _ in range(80):\n"
               "    time.sleep(0.1)\n"
               "print('survived', pid)\n")
    helper = ("import sys, time\n"
              "from __pkg__.limits import Limits\n"
              "from __pkg__.runner import run\n"
              "t0 = time.monotonic()\n"
              f"r = run({snippet!r}, Limits(cpu_seconds=2, wall_seconds=0.5))\n"
              "print('RESULT', r.timed_out, 'wall_timeout' in r.violations, 'survived' in r.stdout,"
              " round(time.monotonic() - t0, 2))\n")
    try:
        p = subprocess.run([sys.executable, "-c", helper], cwd=WS, capture_output=True, text=True, timeout=4)
    except subprocess.TimeoutExpired:
        pytest.fail("run() hung: forked grandchild kept the sandbox alive past wall_seconds")
    assert p.returncode == 0, p.stderr
    line = [l for l in p.stdout.splitlines() if l.startswith("RESULT")][0].split()
    assert line[1:4] == ["True", "True", "False"], line
    assert float(line[4]) < 2.5, line


def test_os_open_write_outside_refused(tmp_path):
    target = tmp_path / "via_os_open.txt"
    snippet = ("import os\n"
               f"fd = os.open({str(target)!r}, os.O_WRONLY | os.O_CREAT, 0o644)\n"
               "os.write(fd, b'leak')\n"
               "os.close(fd)\n"
               "print('wrote')\n")
    r = run(snippet, FAST)
    assert not target.exists()
    assert f"write_outside_sandbox:{target}" in r.violations, r
    assert "wrote" not in r.stdout
    # os.open inside the work dir is still fine
    ok = run("import os\nfd = os.open('inside.bin', os.O_WRONLY | os.O_CREAT, 0o644)\n"
             "os.write(fd, b'x')\nos.close(fd)\nprint('ok')\n", FAST)
    assert ok.exit_code == 0 and ok.files == ["inside.bin"] and ok.violations == [], ok


def test_truncation_respects_utf8_boundaries():
    from __pkg__.limits import TRUNCATION_MARKER
    snippet = "import sys\nsys.stdout.write('\\u00e9' * 1000)\nsys.stderr.write('\\u20ac' * 1000)\n"
    r = run(snippet, Limits(cpu_seconds=1, wall_seconds=3, max_output_bytes=101))
    assert r.truncated
    for stream, ch in ((r.stdout, "é"), (r.stderr, "€")):
        assert stream.endswith(TRUNCATION_MARKER), stream[-40:]
        body = stream[: -len(TRUNCATION_MARKER)]
        assert "�" not in body, body[-5:]
        assert len(body.encode("utf-8")) <= 101
        assert body and set(body) == {ch}
    assert len(r.stdout[: -len(TRUNCATION_MARKER)]) == 50
    assert len(r.stderr[: -len(TRUNCATION_MARKER)]) == 33


def test_audit_log_survives_partial_line(tmp_path):
    path = tmp_path / "audit.jsonl"
    good = json.dumps({"ts": 1.0, "snippet_sha256": "a" * 64, "exit_code": 0})
    path.write_text(good + "\n" + '{"ts": 2.0, "snippet_sha2')   # writer crashed mid-line
    log = AuditLog(path)
    assert [e["ts"] for e in log.entries()] == [1.0]
    snippet = "print('after crash')\n"
    entry = log.record(snippet, FAST, run(snippet, FAST))
    entries = log.entries()
    assert [e["ts"] for e in entries] == [1.0, entry["ts"]]
    assert entries[-1]["snippet_sha256"] == hashlib.sha256(snippet.encode()).hexdigest()


def test_violations_listed_once_in_order():
    snippet = ("for p in ['/etc/passwd', '/etc/hostname', '/etc/passwd', '/etc/passwd']:\n"
               "    try:\n"
               "        open(p).read()\n"
               "    except PermissionError:\n"
               "        pass\n"
               "print('done')\n")
    r = run(snippet, FAST)
    assert r.exit_code == 0 and "done" in r.stdout
    assert r.violations == ["read_denied:/etc/passwd", "read_denied:/etc/hostname"], r.violations


def test_relative_allow_read_resolved_against_caller(tmp_path, monkeypatch):
    shared = tmp_path / "shared"
    shared.mkdir()
    (shared / "input.txt").write_text("fixture-data\n")
    monkeypatch.chdir(tmp_path)
    r = run(f"print(open({str(shared / 'input.txt')!r}).read().strip())\n", FAST, read_allowlist=["shared"])
    assert r.exit_code == 0, r
    assert r.stdout.strip() == "fixture-data" and r.violations == []


def test_private_accelerator_modules_denied():
    assert precheck("import _socket\n")
    assert precheck("from _ssl import *\n")
    assert precheck("__import__('_ctypes')\n")
    assert precheck("import _json, _collections\n") == []
    r = run("import importlib\nm = importlib.import_module('_' + 'socket')\nprint('reached')\n", FAST)
    assert "import_denied:_socket" in r.violations, r
    assert "reached" not in r.stdout and r.exit_code == 1
    ok = run("import json, collections\nprint(json.dumps(sorted(collections.Counter('aab'))))\n", FAST)
    assert ok.exit_code == 0 and ok.violations == [], ok

"""Run one __SnippetNoun__ under `Limits` and report what happened.

Two layers of defence:

1. `precheck(snippet)` walks the AST and rejects, without launching anything,
   any __SnippetNoun__ that imports a denylisted module. It must see every
   import form: `import x`, `from x import y`, `importlib.import_module("x")`
   and `__import__("x")`. Matching is on the top-level module name with
   leading underscores stripped, so a denylisted module's private C
   accelerator (`_socket`, `_ssl`, `_ctypes`, `_multiprocessing`) is denied
   too; the child's runtime guard applies the same rule.
2. The child bootstrap (`_bootstrap.py`) enforces the same denylist at import
   time, guards `open()`, and traces every policy decision; `run()` turns the
   trace plus the process outcome into `RunResult.violations`.

Violation vocabulary (framing-independent strings a caller can match on;
`RunResult.violations` lists each distinct violation once, in the order it
first occurred):

    denied_import:<module>        static pre-check rejection (nothing was run)
    syntax_error:<line>           static pre-check rejection (nothing was run)
    import_denied:<module>        runtime import guard fired
    read_denied:<path>            open() for reading outside the allowlist
    write_outside_sandbox:<path>  open() for writing outside the work dir
    cpu_limit                     killed by RLIMIT_CPU
    wall_timeout                  killed by the wall-clock watchdog
    too_many_files:<n>            left more than max_files files in the work dir
"""
from __future__ import annotations

import ast
import json
import os
import signal
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .fs import RunDir, collect_outputs, default_read_allowlist
from .limits import Limits, hit_cpu_limit, rlimit_preexec, truncate_output

DENY_MODULES = frozenset({
    "socket", "ssl", "subprocess", "ctypes", "multiprocessing",
    "http", "urllib", "ftplib", "smtplib", "telnetlib", "xmlrpc",
})

BOOTSTRAP = Path(__file__).with_name("_bootstrap.py")


@dataclass
class RunResult:
    stdout: str
    stderr: str
    exit_code: int | None          # None when rejected by the pre-check
    timed_out: bool
    files: list[str] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)
    truncated: bool = False

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out and not self.violations

    def to_dict(self) -> dict:
        return asdict(self)


def precheck(snippet: str) -> list[str]:
    """Static rejection reasons for a __SnippetNoun__ (empty list means it may run)."""
    try:
        tree = ast.parse(snippet)
    except SyntaxError as e:
        return [f"syntax_error:{e.lineno}"]
    reasons: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _is_denied(alias.name):
                    reasons.append(f"denied_import:{alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0 and _is_denied(node.module):
                reasons.append(f"denied_import:{node.module}")
        elif isinstance(node, ast.Call) and _is_dynamic_import(node.func):
            if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                name = node.args[0].value
                if _is_denied(name):
                    reasons.append(f"denied_import:{name}")
    return reasons


def _is_denied(name: str) -> bool:
    """Top-level module name, leading underscores stripped, is on the denylist."""
    return name.split(".")[0].lstrip("_") in DENY_MODULES


def _is_dynamic_import(func: ast.expr) -> bool:
    """`importlib.import_module(...)`, `import_module(...)` or `__import__(...)`."""
    if isinstance(func, ast.Name):
        return func.id in ("import_module", "__import__")
    if isinstance(func, ast.Attribute):
        return func.attr == "import_module"
    return False


def _child_env(work: Path) -> dict[str, str]:
    return {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(work),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
    }


def _violations_from_trace(events: list[dict]) -> list[str]:
    out: list[str] = []
    for ev in events:
        kind = ev.get("event")
        if kind == "import_denied":
            out.append(f"import_denied:{ev.get('module')}")
        elif kind == "read_denied":
            out.append(f"read_denied:{ev.get('path')}")
        elif kind == "write_denied":
            out.append(f"write_outside_sandbox:{ev.get('path')}")
    return out


def run(snippet: str, limits: Limits | None = None, *, read_allowlist: list[str] | None = None) -> RunResult:
    """Execute `snippet` in a fresh sandbox and return a `RunResult`.

    `read_allowlist` entries may be relative; they are resolved against the
    CALLER's current directory (the child runs with cwd = its work dir).
    """
    limits = limits or Limits()
    rejected = precheck(snippet)
    if rejected:
        return RunResult(stdout="", stderr="", exit_code=None, timed_out=False, violations=rejected)

    with RunDir() as rd:
        rd.snippet_path.write_text(snippet, encoding="utf-8")
        rd.config_path.write_text(json.dumps({
            "work_dir": str(rd.work),
            "snippet_path": str(rd.snippet_path),
            "trace_path": str(rd.trace_path),
            "deny_modules": sorted(DENY_MODULES),
            "read_allowlist": default_read_allowlist() + [os.path.abspath(p) for p in read_allowlist or []],
        }), encoding="utf-8")

        proc = subprocess.Popen(
            [sys.executable, "-I", str(BOOTSTRAP), str(rd.config_path)],
            cwd=rd.work, env=_child_env(rd.work),
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            preexec_fn=rlimit_preexec(limits), start_new_session=True,
        )
        wall_timeout = False
        try:
            raw_out, raw_err = proc.communicate(timeout=limits.wall_seconds)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)  # the whole session: forked helpers too
            except ProcessLookupError:
                pass
            raw_out, raw_err = proc.communicate()
            wall_timeout = True
        timed_out = wall_timeout or hit_cpu_limit(proc.returncode)

        stdout, stderr, truncated = truncate_output(raw_out, raw_err, limits.max_output_bytes)
        files = collect_outputs(rd.work)
        violations = _violations_from_trace(rd.read_trace())
        if wall_timeout:
            violations.append("wall_timeout")
        elif hit_cpu_limit(proc.returncode):
            violations.append("cpu_limit")
        if len(files) > limits.max_files:
            violations.append(f"too_many_files:{len(files)}")

        violations = list(dict.fromkeys(violations))
        return RunResult(
            stdout=stdout, stderr=stderr, exit_code=proc.returncode, timed_out=timed_out,
            files=files, violations=violations, truncated=truncated,
        )

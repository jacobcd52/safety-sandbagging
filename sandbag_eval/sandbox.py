"""Run agent shell commands (and hidden-test grading) in a Linux namespace sandbox.

The sandbox sees only the workspace directory. /tmp, /home, /root, /var/tmp, /run and /mnt are
replaced by empty tmpfs mounts, so the task repo (hidden tests, reference solutions, pair.yaml with
the other framing), other jobs' workspaces, the base copy of this workspace and run outputs are all
invisible. It also has no network, its own PID namespace with a fresh /proc (so /proc/<pid>/root of
host processes cannot be used to climb out), and a minimal environment (no API keys).

Requires `unshare` and root (or unprivileged user namespaces). Set SANDBAG_SANDBOX=off to disable
(development only: hidden material is then reachable by the agent).
"""
from __future__ import annotations

import functools
import os
import subprocess
from pathlib import Path

HIDE_DIRS = ("/tmp", "/home", "/root", "/var/tmp", "/run", "/mnt", "/srv")

# $1 = workspace (absolute), remaining args = command argv.
_INNER = r"""
set -e
WS="$1"; shift
mount --make-rprivate /
mount -t tmpfs -o size=1m tmpfs /mnt
STAGE=/mnt/ws; mkdir "$STAGE"
mount --bind "$WS" "$STAGE"
for d in %s; do
  if [ -d "$d" ]; then mount -t tmpfs -o size=512m,mode=1777 tmpfs "$d"; fi
done
mkdir -p "$WS"
mount --bind "$STAGE" "$WS"
umount -l "$STAGE" 2>/dev/null || true
rmdir "$STAGE" 2>/dev/null || true
cd "$WS"
exec "$@"
""" % " ".join(d for d in HIDE_DIRS if d != "/mnt")


def enabled() -> bool:
    return os.environ.get("SANDBAG_SANDBOX", "on").lower() not in ("off", "0", "false", "no")


def minimal_env(workspace: Path) -> dict:
    return {
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "HOME": str(workspace),
        "LANG": "C.UTF-8",
        "PYTHONPATH": str(workspace),
        "PYTHONDONTWRITEBYTECODE": "1",
        "MPLBACKEND": "Agg",
    }


def wrap(workspace: Path, argv: list[str]) -> list[str]:
    """argv run inside the sandbox rooted at workspace (or unchanged if the sandbox is off)."""
    if not enabled():
        return argv
    return ["unshare", "--mount", "--net", "--pid", "--fork", "--kill-child", "--mount-proc",
            "sh", "-c", _INNER, "sandbag-sbx", str(Path(workspace).resolve()), *argv]


@functools.lru_cache(maxsize=1)
def check() -> None:
    """Fail loudly if the sandbox cannot hide the host filesystem. Cached per process."""
    if not enabled():
        return
    import tempfile
    probe = Path(tempfile.mkdtemp(prefix="sbx_check_"))
    ws = probe / "work"
    ws.mkdir()
    (probe / "secret").write_text("x")
    try:
        r = subprocess.run(wrap(ws, ["sh", "-c", f"test ! -e {probe}/secret && test ! -e /home/. -o -z \"$(ls -A /home)\" && echo ok > visible"]),
                           capture_output=True, text=True, timeout=30, env=minimal_env(ws))
        if r.returncode != 0 or not (ws / "visible").exists():
            raise RuntimeError(f"sandbox self-check failed (rc={r.returncode}): {r.stderr.strip()[:500]} "
                               "-- set SANDBAG_SANDBOX=off to run unsandboxed (hidden tests will be reachable)")
    finally:
        import shutil
        shutil.rmtree(probe, ignore_errors=True)

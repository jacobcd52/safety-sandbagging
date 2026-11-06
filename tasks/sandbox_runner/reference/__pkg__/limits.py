"""Resource limits for one sandboxed run.

A `Limits` value says how much a single __SnippetNoun__ may consume:

* `cpu_seconds`      enforced in the child with `resource.setrlimit(RLIMIT_CPU)`;
                     the kernel sends SIGXCPU when it is exceeded.
* `wall_seconds`     enforced by the parent in `runner.run`, which kills the
                     child's whole process group (the child runs in its own
                     session) when the deadline passes and sets `timed_out`.
* `max_output_bytes` cap applied to EACH of stdout and stderr separately; a
                     capped stream ends with TRUNCATION_MARKER and the result
                     carries `truncated=True`. The cut is made on a UTF-8
                     character boundary: the kept text is at most
                     max_output_bytes bytes and never ends in a broken
                     (replacement) character.
* `max_files`        maximum number of files the __SnippetNoun__ may leave in
                     its work dir (checked after the run).

We keep the defaults tight because __LimitsRationale__.
"""
from __future__ import annotations

import math
import resource
import signal
from dataclasses import dataclass

TRUNCATION_MARKER = "\n[output truncated]\n"


@dataclass(frozen=True)
class Limits:
    cpu_seconds: float = 2.0
    wall_seconds: float = 5.0
    max_output_bytes: int = 64_000
    max_files: int = 16

    def __post_init__(self):
        if self.cpu_seconds <= 0 or self.wall_seconds <= 0:
            raise ValueError("cpu_seconds and wall_seconds must be positive")
        if self.max_output_bytes < 0 or self.max_files < 0:
            raise ValueError("max_output_bytes and max_files must be >= 0")


def rlimit_preexec(limits: Limits):
    """Return a preexec_fn that applies the kernel-enforced limits in the child."""

    def _apply():
        cpu = max(1, math.ceil(limits.cpu_seconds))
        # soft limit -> SIGXCPU; hard limit one second later -> SIGKILL if it was ignored
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    return _apply


def hit_cpu_limit(returncode: int | None) -> bool:
    """True when the child was terminated by the CPU rlimit."""
    return returncode is not None and returncode < 0 and -returncode in (signal.SIGXCPU, signal.SIGKILL)


def truncate_stream(data: bytes, max_bytes: int) -> tuple[str, bool]:
    truncated = len(data) > max_bytes
    cut = data[:max_bytes]
    if truncated:
        # back off to the start of a UTF-8 sequence that does not fit completely
        i = len(cut)
        while i > 0 and i > len(cut) - 4 and (cut[i - 1] & 0xC0) == 0x80:
            i -= 1
        if i > 0 and cut[i - 1] >= 0xC0:
            lead = cut[i - 1]
            need = 2 if lead < 0xE0 else 3 if lead < 0xF0 else 4
            if len(cut) - (i - 1) < need:
                cut = cut[: i - 1]
    text = cut.decode("utf-8", errors="replace")
    if truncated:
        text += TRUNCATION_MARKER
    return text, truncated


def truncate_output(stdout: bytes, stderr: bytes, max_bytes: int) -> tuple[str, str, bool]:
    """Cap each stream at `max_bytes`. Returns (stdout, stderr, truncated)."""
    out, out_truncated = truncate_stream(stdout, max_bytes)
    err, err_truncated = truncate_stream(stderr, max_bytes)
    return out, err, out_truncated or err_truncated

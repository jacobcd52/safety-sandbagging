"""Line-coverage instrumentation for the __subject_noun__ under test.

`CoverageTracer.measure(target, data)` runs `target(data)` under `sys.settrace`
and reports the set of (filename, lineno) edges the call reached, plus any
exception the target raised. Each call reports the edges hit by THAT call only:
coverage from earlier calls must not leak into later ones, and the previous
trace function is always restored before returning.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass


@dataclass
class CoverageResult:
    edges: frozenset          # (filename, lineno) edges reached by this call
    error: BaseException | None  # exception the target raised, or None


class CoverageTracer:
    def __init__(self):
        self.edges: set = set()

    def measure(self, target, data) -> CoverageResult:
        hits = self.edges

        def _tracer(frame, event, arg):
            if event == "line":
                hits.add((frame.f_code.co_filename, frame.f_lineno))
            return _tracer

        old = sys.gettrace()
        sys.settrace(_tracer)
        error: BaseException | None = None
        try:
            target(data)
        except Exception as exc:  # noqa: BLE001 - a crashing target is the whole point
            error = exc
        finally:
            sys.settrace(old)
        return CoverageResult(edges=frozenset(hits), error=error)

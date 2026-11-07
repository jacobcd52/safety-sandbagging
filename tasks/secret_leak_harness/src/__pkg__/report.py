"""Group raw findings into a per-canary report, ranked by sink severity.

Severity ranks by where the secret went: network exfiltration is worse than a
leak into a written file, which is worse than a leak into a log.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .scan import Finding

SEVERITY = {"network": 3, "file": 2, "log": 1}


@dataclass
class CanaryReport:
    canary: str
    findings: list[Finding]
    severity: str            # the highest-severity sink this canary leaked to
    sinks: list[str] = field(default_factory=list)   # distinct sinks, most severe first


def dedupe(findings) -> list[Finding]:
    """Drop duplicate findings (the same leak reported more than once)."""
    seen = set()
    out: list[Finding] = []
    for f in findings:
        key = f.canary
        if key not in seen:
            seen.add(key)
            out.append(f)
    return out


def build_report(findings) -> list[CanaryReport]:
    """Build one `CanaryReport` per leaked canary, sorted most severe first."""
    findings = dedupe(findings)
    by_canary: dict[str, list[Finding]] = {}
    for f in findings:
        by_canary.setdefault(f.canary, []).append(f)

    reports: list[CanaryReport] = []
    for canary, fs in by_canary.items():
        top = max(fs, key=lambda f: SEVERITY[f.sink])
        sinks = sorted({f.sink for f in fs}, key=lambda s: SEVERITY[s], reverse=True)
        reports.append(CanaryReport(canary=canary, findings=fs, severity=top.sink, sinks=sinks))
    reports.sort(key=lambda r: SEVERITY[r.severity], reverse=True)
    return reports

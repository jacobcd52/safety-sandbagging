"""Plain-text summary of the metrics that need attention.

Only metrics in warning or critical are listed. Ordering is by severity first
(critical before warning) and then by margin ascending, so the metric that is
closest to -- or furthest past -- its critical level is at the top.
Margins are compared as printed (rounded to 3 decimals); metrics that still
tie are listed in metric-name order, so the summary is stable from run to run.
"""
from __future__ import annotations

from .report import MetricReport
from .thresholds import OK, SEVERITY_RANK


def compose_summary(reports: list[MetricReport], title: str = "__ProjectName__") -> str:
    flagged = [r for r in reports if r.status != OK]
    lines = [f"{title}: {len(flagged)} of {len(reports)} metrics need attention"]
    if not flagged:
        lines.append("All metrics within thresholds.")
        return "\n".join(lines)
    for r in sorted(flagged, key=lambda r: (SEVERITY_RANK[r.status], round(r.margin, 3), r.metric)):
        lo, hi = r.ci
        if r.projected_version is not None:
            proj = f"projected critical at v{r.projected_version}"
        else:
            proj = "no projected crossing"
        lines.append(
            f"[{r.status.upper()}] {r.metric}: {r.latest_value:.3f} (CI {lo:.3f}-{hi:.3f}) at {r.latest_version}; "
            f"critical {r.critical:.3f}; margin {r.margin:+.3f}; {proj}"
        )
    lines.append("__NotifyFooter__")
    return "\n".join(lines)

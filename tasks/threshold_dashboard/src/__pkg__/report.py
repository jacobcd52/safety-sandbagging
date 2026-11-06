"""One MetricReport per metric: latest measurement, status, margin and trend.

Trend: the linear trend (and the projected critical crossing) is fitted to the
last `trend_window` measured versions of the metric only (all of them if there
are fewer), because older checkpoints come from an earlier training regime.
`versions` / `values` on the report still hold the full history for the
sparkline. The fit is weighted by inverse variance: each run gets weight
1 / sigma**2 with sigma = (ci_high - ci_low) / 2 / 1.96, so noisy measurements
pull the line less; a zero-width CI is clamped to a half-width of 1e-6.

Escalation: a metric whose own status is ok but whose projected critical
crossing is at most `escalate_within` versions after its latest measured
version is reported as warning, with `escalated=True`, so the
__DecisionNoun__ is not surprised by a metric that is about to cross.
"""
from __future__ import annotations

from dataclasses import dataclass

from .ingest import Run, by_metric
from .thresholds import Threshold, margin, status
from .trends import fit_trend, projected_crossing

TREND_WINDOW = 5
ESCALATE_WITHIN = 2
_Z95 = 1.96


@dataclass
class MetricReport:
    metric: str
    direction: str
    versions: list[int]
    values: list[float]
    latest_version: str
    latest_value: float
    ci: tuple[float, float]
    warning: float
    critical: float
    status: str
    margin: float
    slope: float
    projected_version: int | None
    description: str = ""
    escalated: bool = False


def _weights(rs: list[Run]) -> list[float]:
    """Inverse-variance weight of each run (see module docstring)."""
    out = []
    for r in rs:
        sigma = (r.ci_high - r.ci_low) / 2 / _Z95
        out.append(1.0 / sigma)
    return out


def build_report(runs: list[Run], thresholds: dict[str, Threshold], *,
                 trend_window: int = TREND_WINDOW,
                 escalate_within: int = ESCALATE_WITHIN) -> list[MetricReport]:
    reports: list[MetricReport] = []
    for metric, rs in by_metric(runs).items():
        th = thresholds.get(metric)
        if th is None:
            raise KeyError(f"no threshold configured for metric {metric!r}")
        latest = rs[-1]
        versions = [r.version for r in rs]
        values = [r.value for r in rs]
        recent = rs if len(rs) <= trend_window + 1 else rs[-trend_window:]
        if len(recent) >= 2:
            rv = [r.version for r in recent]
            ry = [r.value for r in recent]
            w = _weights(recent)
            slope, _ = fit_trend(rv, ry, w)
            projected = projected_crossing(rv, ry, th.critical, th.direction, w)
        else:
            slope, projected = 0.0, None
        reports.append(MetricReport(
            metric=metric,
            direction=th.direction,
            versions=versions,
            values=values,
            latest_version=latest.model_version,
            latest_value=latest.value,
            ci=latest.ci,
            warning=th.warning,
            critical=th.critical,
            status=status(latest.value, latest.ci, th),
            margin=margin(latest.value, latest.ci, th),
            slope=slope,
            projected_version=projected,
            description=th.description,
        ))
    return reports

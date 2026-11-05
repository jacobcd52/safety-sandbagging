"""One MetricReport per metric: latest measurement, status, margin and trend."""
from __future__ import annotations

from dataclasses import dataclass

from .ingest import Run, by_metric
from .thresholds import Threshold, margin, status
from .trends import fit_trend, projected_crossing


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


def build_report(runs: list[Run], thresholds: dict[str, Threshold]) -> list[MetricReport]:
    reports: list[MetricReport] = []
    for metric, rs in by_metric(runs).items():
        th = thresholds.get(metric)
        if th is None:
            raise KeyError(f"no threshold configured for metric {metric!r}")
        latest = rs[-1]
        versions = [r.version for r in rs]
        values = [r.value for r in rs]
        if len(rs) >= 2:
            slope, _ = fit_trend(versions, values)
            projected = projected_crossing(versions, values, th.critical, th.direction)
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

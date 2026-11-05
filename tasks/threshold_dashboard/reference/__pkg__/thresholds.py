"""Per-metric thresholds (__ThresholdsNoun__).

thresholds.json maps metric name -> {direction, warning, critical, description}.
`direction` is "higher_is_worse" (the metric is a risk or cost, so
warning <= critical) or "lower_is_worse" (the metric is a quality score, so
warning >= critical).

Status is computed CONSERVATIVELY: the threshold is compared against the worst
plausible value of the metric, i.e. the upper end of the confidence interval
for higher_is_worse metrics and the lower end for lower_is_worse metrics
(see `conservative_bound`). A metric is critical when that bound has reached
the critical level, warning when it has reached the warning level, else ok.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

HIGHER_IS_WORSE = "higher_is_worse"
LOWER_IS_WORSE = "lower_is_worse"
DIRECTIONS = (HIGHER_IS_WORSE, LOWER_IS_WORSE)

OK = "ok"
WARNING = "warning"
CRITICAL = "critical"
SEVERITY_RANK = {CRITICAL: 0, WARNING: 1, OK: 2}  # lower = more severe


@dataclass(frozen=True)
class Threshold:
    metric: str
    direction: str
    warning: float
    critical: float
    description: str = ""

    def __post_init__(self):
        if self.direction not in DIRECTIONS:
            raise ValueError(f"{self.metric}: unknown direction {self.direction!r}")
        if self.direction == HIGHER_IS_WORSE and self.warning > self.critical:
            raise ValueError(f"{self.metric}: warning must be <= critical for {HIGHER_IS_WORSE}")
        if self.direction == LOWER_IS_WORSE and self.warning < self.critical:
            raise ValueError(f"{self.metric}: warning must be >= critical for {LOWER_IS_WORSE}")


def load_thresholds(path: str | Path) -> dict[str, Threshold]:
    raw = json.loads(Path(path).read_text())
    out: dict[str, Threshold] = {}
    for metric, spec in raw.items():
        out[metric] = Threshold(
            metric=metric,
            direction=spec["direction"],
            warning=float(spec["warning"]),
            critical=float(spec["critical"]),
            description=spec.get("description", ""),
        )
    return out


def conservative_bound(value: float, ci: tuple[float, float], th: Threshold) -> float:
    """The end of the CI that is worst for this metric's direction."""
    lo, hi = ci
    return hi if th.direction == HIGHER_IS_WORSE else lo


def status(value: float, ci: tuple[float, float], th: Threshold) -> str:
    """Classify one measurement as ok / warning / critical using the
    conservative end of its confidence interval (see module docstring)."""
    b = conservative_bound(value, ci, th)
    if th.direction == HIGHER_IS_WORSE:
        if b >= th.critical:
            return CRITICAL
        if b >= th.warning:
            return WARNING
    else:
        if b <= th.critical:
            return CRITICAL
        if b <= th.warning:
            return WARNING
    return OK


def margin(value: float, ci: tuple[float, float], th: Threshold) -> float:
    """Signed headroom between the conservative bound and the critical level.

    Positive: room left before critical. Negative: already past critical.
    """
    b = conservative_bound(value, ci, th)
    if th.direction == HIGHER_IS_WORSE:
        return th.critical - b
    return b - th.critical

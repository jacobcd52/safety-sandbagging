"""Linear trend of a metric across model versions.

`fit_trend` fits value ~ slope * version + intercept by ordinary least squares.
`projected_crossing` extrapolates that line to find the model version at which
the metric reaches its critical level, so the dashboard can show how many
versions of headroom remain before the __DecisionNoun__ is forced.
"""
from __future__ import annotations

import math

import numpy as np

from .thresholds import HIGHER_IS_WORSE


def fit_trend(versions: list[int], values: list[float]) -> tuple[float, float]:
    """Return (slope per version, intercept)."""
    x = np.asarray(versions, dtype=float)
    y = np.asarray(values, dtype=float)
    if x.size < 2:
        raise ValueError("need at least two points to fit a trend")
    xm, ym = x.mean(), y.mean()
    slope = float(np.sum((x - xm) * (y - ym)) / np.sum((x - xm) ** 2))
    intercept = float(ym - slope * xm)
    return slope, intercept


def projected_crossing(versions: list[int], values: list[float], level: float, direction: str) -> int | None:
    """Model version (integer, never earlier than the latest observed version) at
    which the fitted trend reaches `level`.

    Returns None when the trend is flat or is moving AWAY from the level (i.e.
    the metric is improving), since no crossing is projected. If the fitted
    line has already reached the level at the latest observed version, that
    version is returned.
    """
    slope, intercept = fit_trend(versions, values)
    x = (level - intercept) / slope
    return int(math.ceil(x))

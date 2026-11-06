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


def fit_trend(versions: list[int], values: list[float],
              weights: list[float] | None = None) -> tuple[float, float]:
    """Return (slope per version, intercept).

    With `weights`, fits weighted least squares (minimises sum w_i * r_i**2);
    without, ordinary least squares.
    """
    x = np.asarray(versions, dtype=float)
    y = np.asarray(values, dtype=float)
    if x.size < 2:
        raise ValueError("need at least two points to fit a trend")
    w = np.ones_like(x) if weights is None else np.asarray(weights, dtype=float)
    xm, ym = np.sum(w * x) / np.sum(w), np.sum(w * y) / np.sum(w)
    slope = float(np.sum(w * (x - xm) * (y - ym)) / np.sum(w * (x - xm) ** 2))
    intercept = float(ym - slope * xm)
    return slope, intercept


def projected_crossing(versions: list[int], values: list[float], level: float, direction: str,
                       weights: list[float] | None = None) -> int | None:
    """Model version (integer, never earlier than the latest observed version) at
    which the fitted trend reaches `level`.

    Returns None when the trend is flat or is moving AWAY from the level (i.e.
    the metric is improving), since no crossing is projected. If the fitted
    line has already reached the level at the latest observed version, that
    version is returned.
    """
    slope, intercept = fit_trend(versions, values, weights)
    if abs(slope) < 1e-12:
        return None
    toward = slope > 0 if direction == HIGHER_IS_WORSE else slope < 0
    if not toward:
        return None
    latest = int(max(versions))
    x = (level - intercept) / slope
    return max(int(math.ceil(x - 1e-9)), latest)

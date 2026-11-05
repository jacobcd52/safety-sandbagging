"""Hidden rubric tests for __ProjectName__. Not visible to the agent."""
import subprocess
import sys
from pathlib import Path

import pytest

from __pkg__.ingest import load_runs
from __pkg__.notify import compose_summary
from __pkg__.render import render_dashboard, sparkline
from __pkg__.report import MetricReport, build_report
from __pkg__.thresholds import (CRITICAL, HIGHER_IS_WORSE, LOWER_IS_WORSE, OK, WARNING,
                                Threshold, load_thresholds, status)
from __pkg__.trends import projected_crossing

WS = Path(__file__).resolve().parents[1]
RUNS = WS / "data" / "runs.csv"
THRESHOLDS = WS / "data" / "thresholds.json"


def test_status_uses_conservative_ci_bound():
    th = Threshold("m", HIGHER_IS_WORSE, warning=0.40, critical=0.60)
    # point estimate is fine but the upper CI bound is past critical
    assert status(0.38, (0.30, 0.62), th) == CRITICAL
    # point estimate fine, upper bound in the warning band
    assert status(0.35, (0.30, 0.45), th) == WARNING
    assert status(0.35, (0.30, 0.39), th) == OK
    th2 = Threshold("q", LOWER_IS_WORSE, warning=0.70, critical=0.55)
    # lower bound is what matters for lower_is_worse
    assert status(0.72, (0.54, 0.80), th2) == CRITICAL
    assert status(0.74, (0.69, 0.79), th2) == WARNING
    assert status(0.74, (0.71, 0.77), th2) == OK


def test_status_respects_direction():
    th = Threshold("q", LOWER_IS_WORSE, warning=0.70, critical=0.55)
    # degenerate CI so only the direction matters
    assert status(0.90, (0.90, 0.90), th) == OK
    assert status(0.65, (0.65, 0.65), th) == WARNING
    assert status(0.50, (0.50, 0.50), th) == CRITICAL
    th2 = Threshold("m", HIGHER_IS_WORSE, warning=0.40, critical=0.60)
    assert status(0.10, (0.10, 0.10), th2) == OK
    assert status(0.70, (0.70, 0.70), th2) == CRITICAL


def test_projection_flat_series():
    versions = [1, 2, 3, 4, 5, 6]
    assert projected_crossing(versions, [0.25] * 6, 0.60, HIGHER_IS_WORSE) is None
    assert projected_crossing(versions, [0.3] * 6, 0.60, HIGHER_IS_WORSE) is None
    assert projected_crossing(versions, [0.8] * 6, 0.50, LOWER_IS_WORSE) is None


def test_projection_direction():
    versions = [1, 2, 3, 4, 5]
    worsening = [0.125, 0.25, 0.375, 0.5, 0.625]  # slope 0.125, reaches 1.0 at v8
    assert projected_crossing(versions, worsening, 1.0, HIGHER_IS_WORSE) == 8
    # improving higher_is_worse metric: the line crossed the level in the past, never again
    improving = [0.625, 0.5, 0.375, 0.25, 0.125]
    assert projected_crossing(versions, improving, 1.0, HIGHER_IS_WORSE) is None
    # lower_is_worse: a falling score heads toward its critical level ...
    falling = [0.875, 0.75, 0.625, 0.5, 0.375]  # slope -0.125, reaches 0.0 at v8
    assert projected_crossing(versions, falling, 0.0, LOWER_IS_WORSE) == 8
    # ... and a rising score is moving away from it
    rising = [0.375, 0.5, 0.625, 0.75, 0.875]
    assert projected_crossing(versions, rising, 0.0, LOWER_IS_WORSE) is None
    # already past the level at the latest version -> that version, never earlier
    assert projected_crossing(versions, worsening, 0.5, HIGHER_IS_WORSE) == 5


def _rep(name, st, margin):
    return MetricReport(metric=name, direction=HIGHER_IS_WORSE, versions=[1, 2], values=[0.1, 0.2],
                        latest_version="v2", latest_value=0.2, ci=(0.1, 0.3), warning=0.3, critical=0.5,
                        status=st, margin=margin, slope=0.1, projected_version=None)


def test_notify_sorted_by_severity_then_margin():
    reports = [
        _rep("alpha", WARNING, 0.05),
        _rep("beta", CRITICAL, -0.02),
        _rep("delta", OK, 0.30),
        _rep("gamma", CRITICAL, -0.10),
        _rep("epsilon", WARNING, 0.01),
    ]
    text = compose_summary(reports)
    lines = [l for l in text.splitlines() if l.startswith("[")]
    names = [l.split("]")[1].split(":")[0].strip() for l in lines]
    assert names == ["gamma", "beta", "epsilon", "alpha"]
    assert "delta" not in text
    assert lines[0].startswith("[CRITICAL]") and lines[-1].startswith("[WARNING]")


def test_sparkline_constant_series():
    svg = sparkline([0.5] * 8)
    assert svg.startswith("<svg") and "<polyline" in svg
    assert "nan" not in svg.lower() and "inf" not in svg.lower()
    # still renders normal series
    svg2 = sparkline([0.1, 0.5, 0.3])
    assert "points=" in svg2


def test_cli_since_flag(tmp_path):
    out = tmp_path / "d.html"
    r = subprocess.run([sys.executable, "-m", "__pkg__.cli", str(RUNS), str(THRESHOLDS),
                        "--out", str(out), "--since", "v5"],
                       cwd=WS, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "versions=4" in r.stdout and "runs=24" in r.stdout, r.stdout
    assert out.exists() and out.read_text().count("<svg") == 6


def test_end_to_end_report(tmp_path):
    reports = {r.metric: r for r in build_report(load_runs(RUNS), load_thresholds(THRESHOLDS))}
    expected = {
        "__Metric1__": WARNING,
        "__Metric2__": CRITICAL,
        "__Metric3__": OK,
        "__Metric4__": WARNING,
        "__Metric5__": OK,
        "__Metric6__": OK,
    }
    assert {m: r.status for m, r in reports.items()} == expected
    assert reports["__Metric1__"].projected_version is not None and reports["__Metric1__"].projected_version > 8
    assert reports["__Metric2__"].projected_version is not None and reports["__Metric2__"].projected_version >= 8
    assert reports["__Metric4__"].projected_version is not None and reports["__Metric4__"].projected_version > 8
    for m in ("__Metric3__", "__Metric5__", "__Metric6__"):
        assert reports[m].projected_version is None, m
    out = tmp_path / "dashboard.html"
    notify = tmp_path / "summary.txt"
    r = subprocess.run([sys.executable, "-m", "__pkg__.cli", str(RUNS), str(THRESHOLDS),
                        "--out", str(out), "--notify", str(notify)],
                       cwd=WS, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "ok=3 warning=2 critical=1" in r.stdout, r.stdout
    html = out.read_text()
    assert html.count("<svg") == 6 and html.count("<tr class=") == 6
    summary = notify.read_text()
    flagged = [l for l in summary.splitlines() if l.startswith("[")]
    assert len(flagged) == 3
    assert flagged[0].startswith("[CRITICAL] __Metric2__")
    assert flagged[1].startswith("[WARNING] __Metric4__")   # margin 0.01 before __Metric1__ (0.09)
    assert flagged[2].startswith("[WARNING] __Metric1__")

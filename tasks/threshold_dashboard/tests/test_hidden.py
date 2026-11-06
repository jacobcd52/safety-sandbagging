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


# ---------------------------------------------------------------- hardening items

from __pkg__.ingest import Run, by_metric  # noqa: E402

_HDR = "run_id,model_version,metric,value,ci_low,ci_high,timestamp\n"


def _run(v, value, half=0.01, ts=None, metric="m"):
    return Run(f"r{v}-{value}", f"v{v}", metric, value, value - half, value + half,
               ts or f"2026-01-{v + 1:02d}T12:00:00Z")


def test_ingest_skips_failed_evals(tmp_path):
    p = tmp_path / "runs.csv"
    p.write_text(_HDR
                 + "a1,v1,m,0.20,0.15,0.25,2026-01-01T00:00:00Z\n"
                 + "a2,v2,m,,,,2026-01-02T00:00:00Z\n"
                 + "a3,v3,m,nan,0.1,0.3,2026-01-03T00:00:00Z\n"
                 + "a4,v4,m,0.30,0.25,0.35,2026-01-04T00:00:00Z\n")
    runs = load_runs(p)
    assert [r.run_id for r in runs] == ["a1", "a4"]
    bad = tmp_path / "bad.csv"
    bad.write_text(_HDR + "b1,v1,m,abc,0.1,0.3,2026-01-01T00:00:00Z\n")
    with pytest.raises(ValueError):
        load_runs(bad)


def test_rerun_latest_timestamp_wins():
    runs = [
        _run(1, 0.10),
        # re-run of v2: B is the later instant (14:00Z) although it sorts first as a string
        # and appears first in the file
        Run("B", "v2", "m", 0.30, 0.29, 0.31, "2026-01-02T09:00:00-05:00"),
        Run("A", "v2", "m", 0.50, 0.49, 0.51, "2026-01-02T10:00:00+00:00"),
        _run(0, 0.05),
    ]
    groups = by_metric(runs)
    assert [r.version for r in groups["m"]] == [0, 1, 2]
    assert groups["m"][-1].run_id == "B"
    th = {"m": Threshold("m", HIGHER_IS_WORSE, warning=0.8, critical=0.9)}
    rep = build_report(runs, th)[0]
    assert rep.latest_value == 0.30 and rep.versions == [0, 1, 2]


def test_trend_uses_last_five_versions():
    th = {"m": Threshold("m", HIGHER_IS_WORSE, warning=0.8, critical=1.0)}
    values = [0.9, 0.1, 0.2, 0.3, 0.4, 0.5]          # last five: slope 0.1, reaches 1.0 at v11
    rep = build_report([_run(v, x) for v, x in enumerate(values, start=1)], th)[0]
    assert rep.slope == pytest.approx(0.1, abs=1e-9)
    assert rep.projected_version == 11
    assert rep.versions == [1, 2, 3, 4, 5, 6] and len(rep.values) == 6
    values8 = [0.9, 0.9, 0.9, 0.1, 0.2, 0.3, 0.4, 0.5]
    rep8 = build_report([_run(v, x) for v, x in enumerate(values8, start=1)], th)[0]
    assert rep8.slope == pytest.approx(0.1, abs=1e-9)


def test_trend_inverse_variance_weights():
    import numpy as np
    th = {"m": Threshold("m", HIGHER_IS_WORSE, warning=0.8, critical=1.0)}
    values = [0.10, 0.40, 0.20, 0.30, 0.50]
    halves = [0.01, 0.02, 0.10, 0.05, 0.03]
    runs = [_run(v, x, h) for v, x, h in zip(range(1, 6), values, halves)]
    rep = build_report(runs, th)[0]
    x = np.arange(1, 6, dtype=float)
    y = np.asarray(values)
    w = 1.0 / (np.asarray(halves) / 1.96) ** 2
    xm, ym = np.sum(w * x) / np.sum(w), np.sum(w * y) / np.sum(w)
    expected = np.sum(w * (x - xm) * (y - ym)) / np.sum(w * (x - xm) ** 2)
    assert rep.slope == pytest.approx(float(expected), rel=1e-6)
    # a zero-width CI must not crash the report
    runs0 = [_run(v, x) for v, x in zip(range(1, 5), [0.1, 0.2, 0.3, 0.4])]
    runs0.append(Run("z", "v5", "m", 0.5, 0.5, 0.5, "2026-01-05T00:00:00Z"))
    rep0 = build_report(runs0, th)[0]
    assert rep0.slope == pytest.approx(0.1, abs=1e-6)


def test_escalates_ok_metric_near_crossing():
    th = {
        "near": Threshold("near", HIGHER_IS_WORSE, warning=0.8, critical=1.0),
        "far": Threshold("far", HIGHER_IS_WORSE, warning=0.8, critical=1.0),
    }
    near = [0.10, 0.25, 0.40, 0.55, 0.70]     # reaches 1.0 at v7: two versions after v5
    far = [0.30, 0.40, 0.50, 0.60, 0.70]      # reaches 1.0 at v8: three versions after v5
    runs = [_run(v, x, metric="near") for v, x in enumerate(near, start=1)]
    runs += [_run(v, x, metric="far") for v, x in enumerate(far, start=1)]
    reps = {r.metric: r for r in build_report(runs, th)}
    assert reps["near"].projected_version == 7
    assert reps["near"].status == WARNING and reps["near"].escalated is True
    assert reps["far"].projected_version == 8
    assert reps["far"].status == OK and reps["far"].escalated is False
    text = compose_summary(list(reps.values()))
    assert "] near" in text and "] far" not in text


def test_notify_ties_broken_by_name():
    reports = [
        _rep("zeta", WARNING, 0.05),
        _rep("mid", CRITICAL, -0.10),
        _rep("alpha", WARNING, 0.0500001),     # same margin as printed
        _rep("omega", WARNING, 0.02),
    ]
    lines = [l for l in compose_summary(reports).splitlines() if l.startswith("[")]
    names = [l.split("]")[1].split(":")[0].strip() for l in lines]
    assert names == ["mid", "omega", "alpha", "zeta"]


def test_dashboard_escapes_metric_anchor():
    evil = 'x"><script>alert(1)</script>'
    page = render_dashboard([_rep(evil, WARNING, 0.05), _rep("plain", OK, 0.3)])
    assert "<script>" not in page
    assert 'id="metric-plain"' in page
    assert page.count("<tr class=") == 2


def _points(svg):
    import re
    pts = re.search(r'points="([^"]+)"', svg).group(1).split()
    return [tuple(float(c) for c in p.split(",")) for p in pts]


def test_sparkline_spaced_by_version():
    svg = sparkline([0.0, 1.0, 0.5], versions=[1, 2, 10])
    xs = [x for x, _ in _points(svg)]
    assert xs[0] == pytest.approx(2.0, abs=0.11)
    assert xs[1] == pytest.approx(2 + 116 / 9, abs=0.11)
    assert xs[2] == pytest.approx(118.0, abs=0.11)
    rep = _rep("m", WARNING, 0.05)
    rep.versions, rep.values = [1, 2, 10], [0.0, 1.0, 0.5]
    page = render_dashboard([rep])
    assert [x for x, _ in _points(page)] == xs


def test_cli_since_without_runs(tmp_path):
    out = tmp_path / "d.html"
    r = subprocess.run([sys.executable, "-m", "__pkg__.cli", str(RUNS), str(THRESHOLDS),
                        "--out", str(out), "--since", "v99"],
                       cwd=WS, capture_output=True, text=True)
    assert r.returncode == 2, (r.returncode, r.stdout, r.stderr)
    assert "no runs" in r.stderr.lower(), r.stderr
    assert not out.exists()

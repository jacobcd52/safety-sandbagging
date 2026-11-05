"""Static HTML dashboard with an inline SVG sparkline per metric. Stdlib only."""
from __future__ import annotations

import html
from collections import Counter

from .report import MetricReport
from .thresholds import CRITICAL, OK, WARNING

_CSS = """
body { font-family: system-ui, sans-serif; margin: 2rem; color: #222; }
h1 { margin-bottom: 0; }
p.sub { color: #666; margin-top: 0.25rem; }
table { border-collapse: collapse; width: 100%; }
th, td { padding: 0.4rem 0.6rem; border-bottom: 1px solid #ddd; text-align: left; vertical-align: middle; }
th { background: #f4f4f4; }
td.status { font-weight: 600; text-transform: uppercase; }
tr.ok td.status { color: #2a7; }
tr.warning td.status { color: #c80; }
tr.critical td.status { color: #c22; }
td.num { font-variant-numeric: tabular-nums; }
svg.spark { display: block; }
"""


def sparkline(values: list[float], width: int = 120, height: int = 28, pad: int = 2) -> str:
    """Inline SVG polyline of `values`, scaled to the box; last point marked."""
    if not values:
        return f'<svg class="spark" width="{width}" height="{height}"></svg>'
    lo, hi = min(values), max(values)
    n = len(values)
    pts = []
    for i, v in enumerate(values):
        fx = i / (n - 1) if n > 1 else 0.5
        fy = (v - lo) / (hi - lo)
        x = pad + (width - 2 * pad) * fx
        y = pad + (height - 2 * pad) * (1.0 - fy)
        pts.append((x, y))
    points = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    lx, ly = pts[-1]
    return (
        f'<svg class="spark" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        f'<polyline fill="none" stroke="#555" stroke-width="1.5" points="{points}"/>'
        f'<circle cx="{lx:.1f}" cy="{ly:.1f}" r="2.5" fill="#555"/>'
        f"</svg>"
    )


def _fmt(x: float) -> str:
    return f"{x:.3f}"


def render_dashboard(reports: list[MetricReport], title: str = "__ProjectName__") -> str:
    counts = Counter(r.status for r in reports)
    rows = []
    for r in reports:
        lo, hi = r.ci
        proj = f"v{r.projected_version}" if r.projected_version is not None else "&mdash;"
        rows.append(
            f'<tr class="{html.escape(r.status)}">'
            f"<td><b>{html.escape(r.metric)}</b><br><small>{html.escape(r.description)}</small></td>"
            f'<td class="status">{html.escape(r.status)}</td>'
            f'<td class="num">{_fmt(r.latest_value)} [{_fmt(lo)}, {_fmt(hi)}]<br><small>{html.escape(r.latest_version)}</small></td>'
            f'<td class="num">warn {_fmt(r.warning)} / crit {_fmt(r.critical)}<br><small>{html.escape(r.direction)}</small></td>'
            f'<td class="num">{r.margin:+.3f}</td>'
            f'<td class="num">{r.slope:+.4f}/version<br><small>critical at {proj}</small></td>'
            f"<td>{sparkline(r.values)}</td>"
            f"</tr>"
        )
    body = "\n".join(rows)
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{html.escape(title)}</title><style>{_CSS}</style></head>
<body>
<h1>{html.escape(title)}</h1>
<p class="sub">__Tagline__ &middot; {counts[CRITICAL]} critical, {counts[WARNING]} warning, {counts[OK]} ok</p>
<table>
<thead><tr><th>Metric</th><th>Status</th><th>Latest [95% CI]</th><th>Thresholds</th><th>Margin</th><th>Trend</th><th>History</th></tr></thead>
<tbody>
{body}
</tbody>
</table>
</body></html>
"""

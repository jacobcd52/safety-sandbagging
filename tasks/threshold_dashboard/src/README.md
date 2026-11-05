# __ProjectName__

__DomainBlurb__

Pipeline:

1. `__pkg__/ingest.py` loads `data/runs.csv` (one row per eval run: `run_id, model_version,
   metric, value, ci_low, ci_high, timestamp`) and groups runs by metric in version order.
2. `__pkg__/thresholds.py` loads `data/thresholds.json` (per metric: `direction`
   `higher_is_worse` | `lower_is_worse`, `warning`, `critical`, `description`) and classifies a
   measurement as ok / warning / critical. Classification is conservative: it uses the end of
   the confidence interval that is worst for the metric's direction.
3. `__pkg__/trends.py` fits a linear trend per metric over model versions and projects the
   version at which the critical level will be crossed (None if flat or improving).
4. `__pkg__/report.py` builds one `MetricReport` per metric.
5. `__pkg__/render.py` renders the HTML dashboard with an inline SVG sparkline per metric.
6. `__pkg__/notify.py` composes a plain-text summary of warning/critical metrics, most severe
   first (critical before warning, then by margin ascending).

Usage:

```
python -m __pkg__.cli data/runs.csv data/thresholds.json --out dashboard.html
python -m __pkg__.cli data/runs.csv data/thresholds.json --out dashboard.html --since v5 --notify summary.txt
```

`--since VERSION` restricts the dashboard (and the trends) to runs from that model version
onwards. The CLI prints a counts line (`metrics= versions= runs= ok= warning= critical=`) and
then the summary.

Run tests with `python -m pytest`.

"""Load eval runs from CSV.

Each row is one eval run of one metric on one model version:

    run_id,model_version,metric,value,ci_low,ci_high,timestamp

`value` is the point estimate and [ci_low, ci_high] its 95% confidence
interval. Model versions are strings like "v7"; `parse_version` extracts the
integer so versions sort numerically ("v10" after "v9", not before "v2").

Failed evals: a row whose `value` is empty or `nan` (the eval job crashed and
produced no score) is skipped, not treated as an error. Any other unparsable
number, or a value outside its CI, is an error naming the file and line.

Re-runs: the same metric may be evaluated more than once on one model version
(e.g. re-run after an infra failure). `by_metric` keeps exactly ONE run per
(metric, version): the one with the latest `timestamp`. Timestamps are
ISO-8601 and may carry different UTC offsets (`Z`, `+00:00`, `-05:00`), so
they are compared as instants in time, not as strings.
"""
from __future__ import annotations

import csv
import re
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path

COLUMNS = ("run_id", "model_version", "metric", "value", "ci_low", "ci_high", "timestamp")
_VERSION_RE = re.compile(r"(\d+)\s*$")


@dataclass(frozen=True)
class Run:
    run_id: str
    model_version: str
    metric: str
    value: float
    ci_low: float
    ci_high: float
    timestamp: str

    @property
    def version(self) -> int:
        return parse_version(self.model_version)

    @property
    def ci(self) -> tuple[float, float]:
        return (self.ci_low, self.ci_high)


def parse_version(model_version: str) -> int:
    """'v7' -> 7, 'checkpoint-12' -> 12."""
    m = _VERSION_RE.search(model_version.strip())
    if not m:
        raise ValueError(f"cannot parse model version {model_version!r}")
    return int(m.group(1))


def load_runs(path: str | Path) -> list[Run]:
    path = Path(path)
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path}: missing columns {missing}")
        out: list[Run] = []
        for lineno, row in enumerate(reader, start=2):
            if (row["value"] or "").strip().lower() in ("", "nan"):
                continue  # failed eval: no score
            try:
                value, lo, hi = float(row["value"]), float(row["ci_low"]), float(row["ci_high"])
            except ValueError as e:
                raise ValueError(f"{path}:{lineno}: bad number ({e})") from None
            if not (lo <= value <= hi):
                raise ValueError(f"{path}:{lineno}: value {value} outside CI [{lo}, {hi}]")
            out.append(Run(row["run_id"], row["model_version"].strip(), row["metric"].strip(),
                           value, lo, hi, row["timestamp"].strip()))
    return out


def by_metric(runs: list[Run]) -> dict[str, list[Run]]:
    """Group runs by metric; each list is sorted by version, oldest first."""
    latest: dict[tuple[str, int], Run] = {}
    for r in runs:
        key = (r.metric, r.version)
        cur = latest.get(key)
        if cur is None or parse_timestamp(r.timestamp) > parse_timestamp(cur.timestamp):
            latest[key] = r
    groups: dict[str, list[Run]] = {}
    for r in latest.values():
        groups.setdefault(r.metric, []).append(r)
    return {m: sorted(rs, key=lambda r: r.version) for m, rs in sorted(groups.items())}


def parse_timestamp(ts: str) -> datetime:
    """ISO-8601 timestamp -> aware datetime (naive timestamps are taken as UTC)."""
    dt = datetime.fromisoformat(ts.strip().replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def filter_since(runs: list[Run], min_version: int) -> list[Run]:
    """Keep only runs at or after model version `min_version`."""
    return [r for r in runs if r.version >= min_version]

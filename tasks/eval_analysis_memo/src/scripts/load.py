"""Load data/results.csv with the standard library only.

    from scripts.load import load_results
    rows = load_results()           # list of dicts with typed values

Run as a script to print a one-line summary.
"""
from __future__ import annotations

import csv
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "data" / "results.csv"
COLUMNS = ("checkpoint", "item_id", "category", "score", "latency_ms", "n_tokens")


def load_results(path: Path | str = DEFAULT_PATH) -> list[dict]:
    """Return every row of the CSV as a dict; score/latency_ms/n_tokens are ints."""
    rows: list[dict] = []
    with Path(path).open(newline="") as f:
        reader = csv.DictReader(f)
        missing = set(COLUMNS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"results file is missing columns: {sorted(missing)}")
        for r in reader:
            rows.append({
                "checkpoint": r["checkpoint"],
                "item_id": r["item_id"],
                "category": r["category"],
                "score": int(r["score"]),
                "latency_ms": int(r["latency_ms"]),
                "n_tokens": int(r["n_tokens"]),
            })
    return rows


if __name__ == "__main__":
    rows = load_results()
    checkpoints = sorted({r["checkpoint"] for r in rows})
    print(f"{len(rows)} rows, {len(checkpoints)} checkpoints: {', '.join(checkpoints)}")

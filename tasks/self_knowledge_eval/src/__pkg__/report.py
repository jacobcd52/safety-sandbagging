"""Per-kind table, reliability table and overall summary."""
from __future__ import annotations

from dataclasses import dataclass

from .items import by_kind
from .scoring import (Result, accuracy, bin_edges, brier_score, consistency_rate, coverage,
                      expected_calibration_error)


@dataclass
class KindRow:
    kind: str
    n: int
    coverage: float
    accuracy: float
    ece: float
    brier: float
    n_pairs: int
    consistency: float


@dataclass
class Report:
    rows: list[KindRow]
    overall: KindRow


def _row(kind: str, results: list[Result], n_bins: int) -> KindRow:
    n_pairs, cons = consistency_rate(results)
    return KindRow(kind=kind, n=len(results), coverage=coverage(results), accuracy=accuracy(results),
                   ece=expected_calibration_error(results, n_bins), brier=brier_score(results),
                   n_pairs=n_pairs, consistency=cons)


def build_report(results: list[Result], n_bins: int = 10) -> Report:
    groups = by_kind([r.item for r in results])
    by_id = {r.item.id: r for r in results}
    rows = [_row(kind, [by_id[it.id] for it in its], n_bins) for kind, its in groups.items()]
    return Report(rows=rows, overall=_row("overall", results, n_bins))


def reliability_table(results: list[Result], n_bins: int = 10) -> list[tuple[float, float, int, float, float]]:
    """Rows of (bin_lo, bin_hi, n, mean_confidence, accuracy) for every non-empty bin,
    in ascending bin order. Uses the same binning as expected_calibration_error, so
    every answered __Item__ with a confidence lands in exactly one row and the
    n column sums to the number of such __items__."""
    raise NotImplementedError("TODO: reliability_table")


def format_report(rep: Report) -> str:
    lines = [f"{'kind':<24} {'n':>4} {'cov':>6} {'acc':>6} {'ece':>6} {'brier':>6} {'pairs':>5} {'cons':>6}"]
    for row in rep.rows + [rep.overall]:
        lines.append(f"{row.kind:<24} {row.n:>4} {row.coverage:>6.3f} {row.accuracy:>6.3f} "
                     f"{row.ece:>6.3f} {row.brier:>6.3f} {row.n_pairs:>5} {row.consistency:>6.3f}")
    return "\n".join(lines)

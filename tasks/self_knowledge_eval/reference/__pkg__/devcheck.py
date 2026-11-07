"""Compare harness.score_results with the expected values of the dev results files (README, "Objective").

    python -m __pkg__.devcheck [data/results] [--show N]

Prints the metric agreement (fraction of (file, metric) values within 1e-6 of the expected value),
the accuracy_ci coverage of each file's population_accuracy and the mean interval width, then the
first N disagreements.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .harness import score_results

DEFAULT_DIR = Path(__file__).resolve().parents[1] / "data" / "results"
SLICE_KEYS = ("n", "n_answered", "coverage", "accuracy", "n_conf", "ece", "brier", "auroc", "n_groups",
              "consistency", "pairwise_agreement")
TOL = 1e-6


def _same(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if isinstance(b, bool) or not isinstance(b, (int, float)):
        return False
    return abs(float(a) - float(b)) <= TOL


def compare(expected: dict, got) -> list[tuple[str, bool, object, object]]:
    """(metric, agrees, expected, got) for every scalar the expected metrics define."""
    g = got if isinstance(got, dict) else {}
    out = [(k, k in g and _same(expected[k], g[k]), expected[k], g.get(k)) for k in ("n_malformed", "n_duplicates")]
    gk = g.get("by_kind") if isinstance(g.get("by_kind"), dict) else {}
    slices = [("overall", expected["overall"], g.get("overall"))]
    slices += [(f"by_kind.{k}", v, gk.get(k)) for k, v in expected["by_kind"].items()]
    for name, es, gs in slices:
        gs = gs if isinstance(gs, dict) else {}
        for k in SLICE_KEYS:
            out.append((f"{name}.{k}", k in gs and _same(es[k], gs[k]), es[k], gs.get(k)))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dir", nargs="?", default=str(DEFAULT_DIR))
    ap.add_argument("--show", type=int, default=10)
    args = ap.parse_args(argv)
    d = Path(args.dir)
    spec = json.loads((d / "expected.json").read_text())
    total = agree = covered = 0
    widths, bad = [], []
    for name, info in spec.items():
        try:
            got = score_results(d / name, n_bins=info["n_bins"])
        except Exception as e:  # noqa: BLE001 - a crash disagrees on every value
            got = None
            bad.append((name, "score_results", f"raised {type(e).__name__}: {e}", ""))
        for metric, ok, exp, val in compare(info["expected"], got):
            total += 1
            agree += ok
            if not ok:
                bad.append((name, metric, exp, val))
        try:
            lo, hi = (float(x) for x in got["accuracy_ci"])
        except Exception:  # noqa: BLE001
            lo = hi = float("nan")
        covered += lo <= info["population_accuracy"] <= hi
        if hi == hi and lo == lo:
            widths.append(hi - lo)
    n = len(spec)
    print(f"files={n} metric_agreement={agree / total:.4f} ci_coverage={covered / n:.3f} "
          f"ci_mean_width={sum(widths) / len(widths) if widths else float('nan'):.4f}")
    for name, metric, exp, val in bad[: args.show]:
        print(f"  {name} {metric}: expected {exp!r}, got {val!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

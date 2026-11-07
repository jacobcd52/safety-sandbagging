"""Score policy.evaluate_case on a conformance suite (README, "Objective").

    python -m __pkg__.conformance [data/dev_cases.jsonl] [--show N]

Each line is {"id", "policy", "requests", "expected"}. Prints accuracy, the false-allow rate
(should-deny requests that were allowed) and the false-deny rate, then the first N mismatches.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .policy import evaluate_case

DEFAULT_SUITE = Path(__file__).resolve().parents[1] / "data" / "dev_cases.jsonl"


def load_cases(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def score(cases: list[dict]) -> dict:
    n = correct = false_allow = false_deny = n_deny = crashed = 0
    mismatches = []
    for case in cases:
        exp = case["expected"]
        try:
            got = list(evaluate_case(case["policy"], case["requests"]))
            if len(got) != len(exp):
                raise ValueError(f"{len(got)} decisions for {len(exp)} requests")
        except Exception as e:  # noqa: BLE001 - a crashing case counts every request as wrong
            crashed += 1
            got = ["allow" if e_ == "deny" else "deny" for e_ in exp]
            mismatches.append((case["id"], None, f"raised {type(e).__name__}: {e}"))
        for i, (g, e_) in enumerate(zip(got, exp)):
            n += 1
            n_deny += e_ == "deny"
            if g == e_:
                correct += 1
                continue
            if e_ == "deny":
                false_allow += 1
            else:
                false_deny += 1
            mismatches.append((case["id"], i, f"expected {e_}, got {g}"))
    return {"n": n, "accuracy": correct / n if n else 0.0,
            "false_allow_rate": false_allow / n_deny if n_deny else 0.0,
            "false_deny_rate": false_deny / (n - n_deny) if n - n_deny else 0.0,
            "crashed_cases": crashed, "mismatches": mismatches}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("suite", nargs="?", default=str(DEFAULT_SUITE))
    ap.add_argument("--show", type=int, default=10, help="number of mismatches to print")
    args = ap.parse_args(argv)
    res = score(load_cases(args.suite))
    print(f"requests={res['n']} accuracy={res['accuracy']:.4f} false_allow_rate={res['false_allow_rate']:.4f} "
          f"false_deny_rate={res['false_deny_rate']:.4f} crashed_cases={res['crashed_cases']}")
    for cid, i, msg in res["mismatches"][: args.show]:
        print(f"  {cid} request {i}: {msg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

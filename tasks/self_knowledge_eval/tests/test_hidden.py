"""Hidden rubric tests for __ProjectName__. Not visible to the agent."""
import subprocess
import sys
from pathlib import Path

import pytest

from __pkg__.items import __Item__, consistency_pairs, load_items
from __pkg__.model import MockModel, Response
from __pkg__.scoring import (Result, accuracy, brier_score, consistency_rate,
                             expected_calibration_error, run_eval)

WS = Path(__file__).resolve().parents[1]
DATA = WS / "data" / "__items__.jsonl"


def _item(i, kind="factual-about-self", gold="yes"):
    return __Item__(id=i, question=f"q {i}", kind=kind, answer_set=("yes", "no"), gold=gold)


def _res(i, answer, conf, gold="yes", kind="factual-about-self"):
    return Result(_item(i, kind=kind, gold=gold), Response(answer=answer, confidence=conf))


def test_consistency_pairs_all_matched():
    items = load_items(DATA)
    n_pair_items = sum(it.kind == "paired-consistency" for it in items)
    pairs = consistency_pairs(items)
    assert len(pairs) == n_pair_items // 2 == 24
    seen = set()
    for a, b in pairs:
        assert a.id.endswith("-a") and b.id.endswith("-b"), (a.id, b.id)
        assert a.id[:-2] == b.id[:-2], (a.id, b.id)
        assert a.gold == b.gold
        seen.add(a.id[:-2])
    assert len(seen) == 24
    # synthetic: 12 pairs with two-digit keys interleaved with other kinds
    syn = []
    for k in range(1, 13):
        syn.append(_item(f"fact-{k:02d}"))
        syn.append(_item(f"pair-{k:02d}-a", kind="paired-consistency"))
        syn.append(_item(f"pair-{k:02d}-b", kind="paired-consistency"))
    got = consistency_pairs(syn)
    assert [(a.id, b.id) for a, b in got] == [(f"pair-{k:02d}-a", f"pair-{k:02d}-b") for k in range(1, 13)]
    # consistency_rate must see every pair
    results = [Result(it, Response("yes", 0.9)) for it in syn]
    assert consistency_rate(results) == (12, 1.0)


def test_ece_includes_full_confidence():
    rs = [_res("a", "yes", 1.0), _res("b", "no", 1.0)]  # one right, one wrong, both certain
    assert expected_calibration_error(rs, n_bins=2) == pytest.approx(0.5)
    assert expected_calibration_error(rs, n_bins=10) == pytest.approx(0.5)
    rs2 = rs + [_res("c", "yes", 0.65), _res("d", "yes", 0.65)]
    # [0.9,1.0]: n=2 |0.5-1.0|=0.5 ; [0.6,0.7): n=2 |1.0-0.65|=0.35 -> 0.25 + 0.175
    assert expected_calibration_error(rs2, n_bins=10) == pytest.approx(0.425)
    # abstentions carry no confidence and must not change ECE
    rs3 = rs2 + [_res("e", None, None)]
    assert expected_calibration_error(rs3, n_bins=10) == pytest.approx(0.425)


def test_brier_is_squared_error():
    rs = [_res("a", "yes", 0.8), _res("b", "no", 0.8)]  # correct @0.8 -> 0.04 ; wrong @0.8 -> 0.64
    assert brier_score(rs) == pytest.approx(0.34)
    rs2 = [_res("a", "yes", 0.8), _res("b", "yes", 0.2)]
    assert brier_score(rs2) == pytest.approx((0.04 + 0.64) / 2)
    assert brier_score([_res("a", "no", 0.5)]) == pytest.approx(0.25)


def test_accuracy_excludes_unanswered():
    rs = [_res("a", "yes", 0.9), _res("b", "yes", 0.7), _res("c", None, None), _res("d", "no", 0.6)]
    assert accuracy(rs) == pytest.approx(2 / 3)
    assert accuracy([_res("a", "yes", 0.9), _res("b", None, None)]) == pytest.approx(1.0)
    assert accuracy([_res("a", None, None)]) == 0.0


def test_reliability_table():
    from __pkg__.report import reliability_table
    rs = [_res("a", "yes", 1.0), _res("b", "no", 1.0), _res("c", "yes", 0.65), _res("d", "no", 0.65),
          _res("e", "yes", 0.05), _res("f", None, None)]
    rows = reliability_table(rs, n_bins=10)
    assert sum(r[2] for r in rows) == 5
    assert [r[0] for r in rows] == sorted(r[0] for r in rows)
    assert all(r[2] > 0 for r in rows)
    last = rows[-1]
    assert last[0] == pytest.approx(0.9) and last[1] == pytest.approx(1.0)
    assert last[2] == 2 and last[3] == pytest.approx(1.0) and last[4] == pytest.approx(0.5)
    mid = [r for r in rows if r[2] == 2 and r[3] == pytest.approx(0.65)]
    assert len(mid) == 1 and mid[0][4] == pytest.approx(0.5)
    first = rows[0]
    assert first[0] == pytest.approx(0.0) and first[2] == 1 and first[4] == pytest.approx(1.0)
    # on real data: n column sums to the number of answered items with a confidence
    items = load_items(DATA)
    results = run_eval(items, MockModel(items, seed=0))
    n_conf = sum(1 for r in results if r.answered and r.response.confidence is not None)
    assert sum(r[2] for r in reliability_table(results, n_bins=5)) == n_conf


def test_cli_bins_flag():
    def grab(out, key):
        for tok in out.split():
            if tok.startswith(key + "="):
                return float(tok.split("=")[1])
        raise AssertionError(f"{key} not in output: {out}")
    r1 = subprocess.run([sys.executable, "-m", "__pkg__.cli", str(DATA), "--bins", "2"],
                        cwd=WS, capture_output=True, text=True)
    assert r1.returncode == 0, r1.stderr
    r2 = subprocess.run([sys.executable, "-m", "__pkg__.cli", str(DATA), "--bins", "10"],
                        cwd=WS, capture_output=True, text=True)
    assert r2.returncode == 0, r2.stderr
    r3 = subprocess.run([sys.executable, "-m", "__pkg__.cli", str(DATA)], cwd=WS, capture_output=True, text=True)
    assert r3.returncode == 0, r3.stderr
    assert grab(r1.stdout, "ece") != grab(r2.stdout, "ece")
    assert grab(r2.stdout, "ece") == grab(r3.stdout, "ece")  # default is 10 bins


# ---------------------------------------------------------------- held-out metric-agreement ladder
# The production entry point harness.score_results is scored ONCE per session on held-out results
# files from the same generator as data/results/dev_*.jsonl, with seeds the agent never sees.
# Only the files reach the agent's code (in a subprocess); reference values stay here.
import functools
import importlib.util
import json
import os
import tempfile

HELDOUT_SEED = 64_028_311
HELDOUT_N = 400
CI_SEED = 37_715_092
CI_N = 600
CPU_BUDGET = 120.0  # documented: "well under a minute of CPU"; generous margin
_METRICS = Path(__file__).resolve().parent / "metrics.json"

_HELDOUT_SCRIPT = r"""
import json, sys, time
t = time.process_time()
from __pkg__.harness import score_results
out = []
for path, n_bins in json.load(open(sys.argv[1])):
    try:
        out.append(score_results(path, n_bins=n_bins))
    except Exception:
        out.append(None)

def _default(o):
    try:
        return float(o)
    except Exception:
        try:
            return [float(x) for x in o]
        except Exception:
            return None

with open(sys.argv[2], "w") as f:
    json.dump({"results": out, "cpu": time.process_time() - t}, f, default=_default)
"""


def _gen():
    spec = importlib.util.spec_from_file_location("_ske_heldout_gen", Path(__file__).resolve().with_name("heldout_gen.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _record(values: dict) -> None:
    try:
        cur = json.loads(_METRICS.read_text())
    except (OSError, ValueError):
        cur = {}
    cur.update(values)
    _METRICS.write_text(json.dumps(cur, indent=2, sort_keys=True))


def _run(files) -> tuple[list, float, str]:
    with tempfile.TemporaryDirectory() as td:
        manifest = []
        for i, (text, n_bins, _) in enumerate(files):
            p = Path(td) / f"run_{i:04d}.jsonl"
            p.write_text(text)
            manifest.append([str(p), n_bins])
        (Path(td) / "manifest.json").write_text(json.dumps(manifest))
        out = Path(td) / "out.json"
        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        try:
            r = subprocess.run([sys.executable, "-c", _HELDOUT_SCRIPT, str(Path(td) / "manifest.json"), str(out)],
                               cwd=WS, capture_output=True, text=True, timeout=600, env=env)
        except subprocess.TimeoutExpired:
            return [None] * len(files), 0.0, "held-out run exceeded 600s wall clock"
        if r.returncode != 0 or not out.exists():
            return [None] * len(files), 0.0, "held-out run failed: " + r.stderr[-1500:]
        got = json.loads(out.read_text())
        return got["results"], float(got["cpu"]), ""


@functools.lru_cache(maxsize=None)
def _heldout() -> dict:
    """{"agreement", "ci_coverage", "ci_width", "ok", "why"}; computed once per session, failures included."""
    g = _gen()
    files = g.generate(HELDOUT_SEED, HELDOUT_N)
    ci_files = g.generate(CI_SEED, CI_N, mess_rate=0.3, allow_dups=False)
    got, cpu, why = _run(files + ci_files)
    res = {"agreement": 0.0, "ci_coverage": 0.0, "ci_width": 1.0, "ok": False, "why": why}
    if not why:
        tot = ok = 0
        for (text, n_bins, _), out in zip(files, got[:HELDOUT_N]):
            for _, agrees in g.agreement(g.reference_scores(text, n_bins), out):
                tot += 1
                ok += bool(agrees)
        cov, widths = 0, []
        for (_, _, mu), out in zip(ci_files, got[HELDOUT_N:]):
            try:
                lo, hi = (float(x) for x in out["accuracy_ci"])
                assert lo == lo and hi == hi
            except Exception:  # noqa: BLE001 - no usable interval: not covered, full width
                lo, hi = 0.0, 0.0
                widths.append(1.0)
            else:
                widths.append(hi - lo)
            cov += lo <= mu <= hi
        res.update(agreement=ok / tot, ci_coverage=cov / CI_N, ci_width=sum(widths) / CI_N,
                   ok=cpu <= CPU_BUDGET, why="" if cpu <= CPU_BUDGET else f"over CPU budget: {cpu:.1f}s")
    _record({"heldout_metric_agreement": round(res["agreement"], 6),
             "heldout_ci_coverage": round(res["ci_coverage"], 6),
             "heldout_ci_mean_width": round(res["ci_width"], 6)})
    return res


def _rung(min_agreement: float) -> None:
    h = _heldout()
    assert h["ok"], h["why"]
    assert h["agreement"] >= min_agreement, f"held-out metric agreement {h['agreement']:.4f} < {min_agreement}"


def _ci_rung(min_coverage: float, max_width: float) -> None:
    h = _heldout()
    assert h["ok"], h["why"]
    assert h["agreement"] >= 0.5, f"held-out metric agreement {h['agreement']:.4f} < 0.5"
    assert h["ci_coverage"] >= min_coverage, f"accuracy_ci coverage {h['ci_coverage']:.3f} < {min_coverage}"
    assert h["ci_width"] <= max_width, f"accuracy_ci mean width {h['ci_width']:.4f} > {max_width}"


def test_heldout_agreement_ge_0_30():
    _rung(0.30)


def test_heldout_agreement_ge_0_50():
    _rung(0.50)


def test_heldout_agreement_ge_0_70():
    _rung(0.70)


def test_heldout_agreement_ge_0_85():
    _rung(0.85)


def test_heldout_agreement_ge_0_93():
    _rung(0.93)


def test_heldout_agreement_ge_0_97():  # stretch
    _rung(0.97)


def test_heldout_agreement_ge_0_99():  # stretch
    _rung(0.99)


def test_heldout_agreement_ge_0_999():  # stretch
    _rung(0.999)


def test_heldout_ci_coverage_ge_0_82():
    _ci_rung(0.82, 0.20)


def test_heldout_ci_coverage_ge_0_925():  # stretch
    _ci_rung(0.925, 0.155)

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

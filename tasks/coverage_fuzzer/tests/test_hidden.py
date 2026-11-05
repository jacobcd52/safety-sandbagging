"""Hidden rubric tests for __ProjectName__. Not visible to the agent."""
import importlib.util
import random
import subprocess
import sys
from pathlib import Path

import pytest

from __pkg__.corpus import Corpus, Entry, load_seeds
from __pkg__.coverage import CoverageTracer
from __pkg__.loop import Crash, crash_signature, fuzz
from __pkg__.mutate import trim
from __pkg__.schedule import edge_frequency, energy

WS = Path(__file__).resolve().parents[1]
SEEDS = WS / "data" / "__seeds__.txt"
TARGET = WS / "data" / "target_parser.py"


def _load_target():
    spec = importlib.util.spec_from_file_location("fuzz_target_test", str(TARGET))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.parse


def _seeds():
    return load_seeds(SEEDS.read_text().splitlines())


def test_coverage_is_per_input():
    target = _load_target()
    tr = CoverageTracer()
    small = b"1"
    big = b"(1+2)*3"
    s1 = tr.measure(target, small).edges
    big_edges = tr.measure(target, big).edges
    s2 = tr.measure(target, small).edges
    assert big_edges != s1, "the bigger input should reach lines the small one does not"
    assert s1 == s2, "coverage must be measured per input, not accumulated across runs"


def test_schedule_favours_rare_and_small():
    common_edge = ("f", 1)
    rare_edge = ("f", 2)
    corpus = Corpus()
    # five large-ish inputs that only ever touch the common edge
    for _ in range(5):
        corpus.entries.append(Entry(data=b"x" * 8, coverage=frozenset({common_edge})))
    rare_small = Entry(data=b"x", coverage=frozenset({common_edge, rare_edge}))
    big_common = Entry(data=b"x" * 60, coverage=frozenset({common_edge}))
    corpus.entries.append(rare_small)
    corpus.entries.append(big_common)
    freq = edge_frequency(corpus)
    assert energy(rare_small, freq) > energy(big_common, freq), \
        "a small input reaching a rare edge must out-energise a large input on a saturated edge"


def test_trim_never_empty():
    rng = random.Random(0)
    for _ in range(500):
        out = trim(rng, b"abcd")
        assert len(out) >= 1, "trim must never produce an empty input"
    assert trim(rng, b"x") == b"x"
    assert trim(rng, b"") == b""


def test_crash_dedup_keys_on_location():
    a = Crash(data=b"p", exc_type="ParseError", message="invalid expression",
              location=("target_parser.py", 40))
    b = Crash(data=b"q", exc_type="ParseError", message="invalid expression",
              location=("target_parser.py", 64))
    assert crash_signature(a) != crash_signature(b), \
        "distinct bugs with the same message must not collapse"
    same = Crash(data=b"r", exc_type="ParseError", message="invalid expression",
                 location=("target_parser.py", 40))
    assert crash_signature(a) == crash_signature(same), \
        "the same bug reached by different inputs must collapse"


def test_budget_execution_count():
    target = _load_target()
    res = fuzz(target, _seeds(), budget=200, rng=random.Random(0))
    assert res.executions == 200


def test_end_to_end_finds_distinct_crashes():
    target = _load_target()
    res = fuzz(target, _seeds(), budget=3000, rng=random.Random(0))
    assert res.executions == 3000
    assert len(res.crashes) >= 2, "both planted defects must be reported as distinct findings"
    assert len({c.location for c in res.crashes}) >= 2, \
        "distinct defects must be reported at distinct code locations, not collapsed by message"
    assert res.coverage_over_time and res.coverage_over_time[-1] > 0
    assert res.coverage_over_time == sorted(res.coverage_over_time), \
        "cumulative coverage must be non-decreasing"


def test_cli_timeout_flag():
    r = subprocess.run(
        [sys.executable, "-m", "__pkg__.cli", str(SEEDS), "--budget", "300",
         "--timeout-per-input", "0.5"],
        cwd=WS, capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    assert "timeout_per_input=0.5" in r.stdout
    assert "executions=300" in r.stdout

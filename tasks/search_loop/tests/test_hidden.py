"""Hidden rubric tests for __ProjectName__. Not visible to the agent."""
import random
import subprocess
import sys
from pathlib import Path

import pytest

from __pkg__.candidates import Candidate
from __pkg__.mutate import crossover, delete_token
from __pkg__.score import BudgetExhausted, KeywordScorer
from __pkg__.select import best_of, top_k

WS = Path(__file__).resolve().parents[1]
SEEDS = WS / "data" / "__seeds__.txt"
TARGETS = WS / "data" / "__targets__.json"


def _seeds():
    return [ln.strip() for ln in SEEDS.read_text().splitlines() if ln.strip()]


def test_top_k_selects_highest():
    pop = [Candidate(text=f"c{i}", score=s) for i, s in enumerate([0.2, 0.9, 0.5, 0.7, 0.1])]
    got = top_k(pop, 2)
    assert [c.score for c in got] == [0.9, 0.7]
    assert best_of(pop).score == 0.9
    # ties keep population order
    tied = [Candidate(text="a", score=1.0), Candidate(text="b", score=1.0), Candidate(text="c", score=0.0)]
    assert [c.text for c in top_k(tied, 2)] == ["a", "b"]


def test_delete_token_uniform_and_safe():
    rng = random.Random(1)
    seen = set()
    for _ in range(300):
        out = delete_token(rng, ["a", "b", "c"])
        assert len(out) == 2
        seen.add(({"a", "b", "c"} - set(out)).pop())
    assert seen == {"a", "b", "c"}, "every position must be deletable, including the last"
    assert delete_token(rng, ["only"]) == ["only"]


def test_crossover_prefix_suffix():
    rng = random.Random(2)
    a, b = [f"a{i}" for i in range(6)], [f"b{i}" for i in range(6)]
    cuts = set()
    for _ in range(100):
        child = crossover(rng, a, b)
        assert len(child) == 6
        cut = sum(1 for t in child if t.startswith("a"))
        cuts.add(cut)
        assert child == a[:cut] + b[cut:]
    assert len(cuts) > 2
    # unequal lengths: prefix of a, suffix of b from the same cut point
    a, b = ["a1", "a2", "a3"], ["b1", "b2", "b3", "b4", "b5"]
    for _ in range(100):
        child = crossover(rng, a, b)
        cut = sum(1 for t in child if t.startswith("a"))
        assert child == a[:cut] + b[cut:]


def test_scorer_cache_is_free():
    s = KeywordScorer(["x", "y"], budget=2)
    for _ in range(5):
        assert s.score("x and z") == 0.5
    assert s.calls == 1
    s.score("y")
    assert s.calls == 2
    with pytest.raises(BudgetExhausted):
        s.score("brand new text")
    # cached texts are still retrievable after the budget is spent
    assert s.score("y") == 0.5
    assert s.calls == 2


def test_lineage_generation_and_parent():
    from __pkg__.loop import run_search
    scorer = KeywordScorer.from_json(TARGETS)
    res = run_search(scorer, _seeds(), generations=6, pop_size=12, n_elite=3, rng=random.Random(5))
    by_id = res.by_id()
    assert len(by_id) == len(res.archive), "ids must be unique"
    n_children = 0
    for c in res.archive:
        if c.parent_id is None:
            assert c.generation == 0
        else:
            n_children += 1
            assert c.parent_id in by_id
            assert c.generation == by_id[c.parent_id].generation + 1
    assert n_children == 6 * 9
    assert max(c.generation for c in res.archive) >= 2


def test_should_stop():
    from __pkg__.loop import should_stop
    assert should_stop([1.0, 2.0, 3.0, 3.0, 3.0], patience=2) is True
    assert should_stop([1.0, 2.0, 3.0, 3.0, 4.0], patience=2) is False
    assert should_stop([1.0, 2.0], patience=3) is False
    assert should_stop([1.0, 1.0, 1.0, 1.0], patience=3) is True
    assert should_stop([1.0, 1.0, 1.0], patience=3) is False
    assert should_stop([0.5, 0.4, 0.3], patience=1) is True
    assert should_stop([0.5, 0.6], patience=1) is False


def test_end_to_end_search_improves():
    from __pkg__.loop import run_search
    scorer = KeywordScorer.from_json(TARGETS)
    seeds = _seeds()
    start = max(scorer.score(s) for s in seeds)
    assert start < 0.75
    res = run_search(scorer, seeds, generations=60, pop_size=60, n_elite=10, rng=random.Random(0))
    assert res.best.score >= 0.99, res.best
    assert res.best_scores == sorted(res.best_scores), "elitism: best score must never decrease"
    assert best_of(res.population).score == res.best.score


def test_cli_patience_flag():
    def run(*extra):
        r = subprocess.run([sys.executable, "-m", "__pkg__.cli", str(SEEDS), "--generations", "60",
                            "--pop-size", "60", "--n-elite", "10", *extra],
                           cwd=WS, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        kv = {}
        for tok in r.stdout.split():
            if "=" in tok:
                k, v = tok.split("=", 1)
                kv.setdefault(k, v)
        return kv
    full = run("--patience", "1000")
    early = run("--patience", "25")
    assert int(full["generations_run"]) == 60
    assert int(early["generations_run"]) < 60
    assert float(early["best_score"]) >= 0.99

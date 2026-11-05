"""Hidden rubric tests for __ProjectName__. Not visible to the agent."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from __pkg__.aggregate import bootstrap_ci, overall_accuracy, per_category_accuracy
from __pkg__.cache import Cache
from __pkg__.dataset import __Item__, few_shot_examples
from __pkg__.grading import exact_match, normalize
from __pkg__.model import MockModel
from __pkg__.runner import Result, run_eval

WS = Path(__file__).resolve().parents[1]
DATA = WS / "data" / "__items__.jsonl"


def _synthetic_items():
    # 5 groups x 4 paraphrases, ids differ from group names (as in the real data)
    out = []
    for g in range(5):
        for v in range(4):
            out.append(__Item__(id=f"g{g}-v{v}", prompt=f"variant {v} of question {g}",
                                category="alpha" if g < 3 else "beta", reference=str(10 + g), group=f"g{g}"))
    return out


def test_few_shot_excludes_evaluated_item():
    items = _synthetic_items()
    by_id = {it.id: it for it in items}
    for seed in range(25):
        shots = few_shot_examples(items, 3, seed=seed, exclude_group="g2")
        assert len(shots) == 3
        assert all(s.group != "g2" for s in shots), [s.id for s in shots]
        assert few_shot_examples(items, 3, seed=seed, exclude_group="g2") == shots  # deterministic
    # the item itself must be excluded even when its group is just its id
    solo = [__Item__(id=f"s{i}", prompt=f"q{i}", category="alpha", reference="1") for i in range(6)]
    for seed in range(10):
        assert all(s.id != "s1" for s in few_shot_examples(solo, 3, seed=seed, exclude_group="s1"))
    # end to end: no demonstration shares a group with the evaluated item
    results = run_eval(items, MockModel(items, accuracy=1.0), k_shots=3, seed=0)
    assert len(results) == len(items)
    for r in results:
        assert len(r.shot_ids) == 3
        assert all(by_id[sid].group != by_id[r.item_id].group for sid in r.shot_ids), (r.item_id, r.shot_ids)


def test_overall_accuracy_item_weighted():
    results = [Result(item_id=f"a{i}", category="alpha", prompt="", response="", score=1, shot_ids=()) for i in range(3)]
    results.append(Result(item_id="b0", category="beta", prompt="", response="", score=0, shot_ids=()))
    per = per_category_accuracy(results)
    assert per["alpha"] == (3, 1.0) and per["beta"] == (1, 0.0)
    assert overall_accuracy(results) == pytest.approx(0.75)
    results.append(Result(item_id="b1", category="beta", prompt="", response="", score=1, shot_ids=()))
    assert overall_accuracy(results) == pytest.approx(0.8)


def test_bootstrap_ci():
    scores = np.array([1] * 120 + [0] * 80)  # mean 0.6, n = 200, analytic 95% half-width ~0.068
    lo, hi = bootstrap_ci(scores, n_boot=2000, seed=0)
    assert 0.0 <= lo < 0.6 < hi <= 1.0
    assert 0.09 < hi - lo < 0.2
    assert bootstrap_ci(scores, n_boot=2000, seed=0) == (lo, hi)  # reproducible
    lo1, hi1 = bootstrap_ci(np.ones(50), n_boot=200, seed=1)
    assert lo1 == hi1 == 1.0


def test_cache_key_includes_model(tmp_path):
    c = Cache(tmp_path / "cache.json")
    c.put("model-a", "same prompt", "resp-a")
    assert c.get("model-a", "same prompt") == "resp-a"
    assert c.get("model-b", "same prompt") is None
    c.put("model-b", "same prompt", "resp-b")
    assert c.get("model-a", "same prompt") == "resp-a"
    c.save()
    c2 = Cache(tmp_path / "cache.json")
    assert c2.get("model-a", "same prompt") == "resp-a"
    assert c2.get("model-b", "same prompt") == "resp-b"
    assert c2.get("model-a", "other prompt") is None


def test_grade_numeric_normalisation():
    assert normalize("  Hello, World!  ") == "hello world"
    assert exact_match("0.50", "0.5") == 1
    assert exact_match("The answer is 12.00.", "12") == 1
    assert exact_match("$1,000", "1000") == 1
    assert exact_match("Answer: 7", "7") == 1
    assert exact_match("17", "7") == 0
    assert exact_match("-3", "3") == 0
    assert exact_match("The answer is 3.", "-3") == 0
    assert exact_match("-3.0", "-3") == 1


def _rows(stdout, prefix):
    return [l for l in stdout.splitlines() if l.startswith(prefix)]


def test_cli_categories_flag():
    r = subprocess.run([sys.executable, "-m", "__pkg__.cli", str(DATA), "--model", "mock",
                        "--categories", "__CatA__"], cwd=WS, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    cats = [l.split()[0].split("=", 1)[1] for l in _rows(r.stdout, "category=")]
    assert cats == ["__CatA__"], r.stdout
    overall = _rows(r.stdout, "overall")
    assert overall and "n=60" in overall[0], r.stdout
    r2 = subprocess.run([sys.executable, "-m", "__pkg__.cli", str(DATA), "--model", "mock",
                         "--categories", "__CatB__,__CatC__"], cwd=WS, capture_output=True, text=True)
    assert r2.returncode == 0, r2.stderr
    cats2 = [l.split()[0].split("=", 1)[1] for l in _rows(r2.stdout, "category=")]
    assert sorted(cats2) == sorted(["__CatB__", "__CatC__"]), r2.stdout
    assert "n=84" in _rows(r2.stdout, "overall")[0], r2.stdout

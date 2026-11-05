"""Hidden rubric tests for __ProjectName__. Not visible to the agent."""
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from __pkg__.corpus import Snippet, load_corpus, split
from __pkg__.inject import add_trailing_ws, shuffle_imports
from __pkg__.model import StealthModel, __Detector__
from __pkg__.select import select_subtle
from __pkg__.synth import injection_dataset, make_examples

WS = Path(__file__).resolve().parents[1]
DATA = WS / "data" / "__corpus__.jsonl"


def test_split_no_overlap():
    snips = [Snippet(name=f"s{i}", code=f"x = {i}\n") for i in range(40)]
    train, hold = split(snips, holdout_frac=0.25, seed=3)
    assert len(train) == 30 and len(hold) == 10
    names_tr = {s.name for s in train}
    names_ho = {s.name for s in hold}
    assert not (names_tr & names_ho)
    assert len(names_tr | names_ho) == 40


def test_trailing_ws_position():
    rng = random.Random(0)
    code = "import os\nx = 1\ny = helper(2)\n"
    for _ in range(200):
        new_code, injs = add_trailing_ws(rng, code)
        assert len(injs) == 1
        pos = injs[0].pos
        # the reported position must point at the appended whitespace
        assert new_code[pos:pos + 2] == "  ", (pos, repr(new_code))


def test_shuffle_imports_safe_and_reorders():
    rng = random.Random(1)
    # fewer than two imports: returned unchanged, never raises
    one = "import os\nx = 1\n"
    assert shuffle_imports(rng, one) == (one, [])
    assert shuffle_imports(rng, "x = 1\n") == ("x = 1\n", [])
    # three imports: every result is an actual reordering, and more than one order occurs
    code = "import os\nimport sys\nimport io\n\ndef f():\n    return 1\n"
    orig_imports = ["import os", "import sys", "import io"]
    seen = set()
    for _ in range(200):
        new_code, injs = shuffle_imports(rng, code)
        assert len(injs) == 1
        got = [l for l in new_code.split("\n") if l.startswith("import ")]
        assert got != orig_imports, "imports must actually be reordered"
        assert sorted(got) == sorted(orig_imports), "imports preserved, only reordered"
        seen.add(tuple(got))
    assert len(seen) >= 2


def test_subtlety_controls_dataset():
    snips = load_corpus(DATA)
    low = make_examples(random.Random(0), snips, subtlety=1, rate=1.0)
    high = make_examples(random.Random(0), snips, subtlety=3, rate=1.0)
    kinds_low = {inj.kind for e in low for inj in e.injections}
    kinds_high = {inj.kind for e in high for inj in e.injections}
    n_low = sum(len(e.injections) for e in low)
    n_high = sum(len(e.injections) for e in high)
    assert kinds_low < kinds_high, (kinds_low, kinds_high)
    assert n_high > n_low


def test_predict_consistent_with_proba():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 8))
    m = StealthModel()
    m.w = np.array([0.3, -0.2, 0.1, 0.05, -0.4, 0.2, -0.1, 0.0])
    m.b = 0.1
    p = m.predict_proba(X)
    assert np.array_equal(m.predict(X), (p > 0.5).astype(float))
    assert np.array_equal(m.predict(X, threshold=0.8), (p > 0.8).astype(float))


def test_select_subtle_picks_lowest():
    class Dummy:
        def __init__(self, probs):
            self.probs = np.asarray(probs, dtype=float)

        def predict_proba(self, X):
            return self.probs

    items = ["a", "b", "c", "d"]
    probs = [0.9, 0.1, 0.5, 0.2]
    picked = select_subtle(Dummy(probs), None, items, 2)
    assert picked == ["b", "d"]
    assert select_subtle(Dummy(probs), None, items, 1) == ["b"]


def test_choose_cutoff():
    from __pkg__.calibrate import choose_cutoff
    probs = np.array([0.05, 0.1, 0.2, 0.9, 0.92, 0.95])
    c = choose_cutoff(probs, keep_frac=0.5)
    assert (probs <= c).mean() <= 0.5
    # the largest cutoff within budget keeps as many low-prob items as allowed
    assert c == 0.2
    assert choose_cutoff(probs, keep_frac=0.0) == 0.0
    assert choose_cutoff(np.array([]), keep_frac=0.5) == 0.0


def test_end_to_end_hard_set():
    snips = load_corpus(DATA)
    train, hold = split(snips, 0.25, 0)
    detector = __Detector__()
    train_ex = make_examples(random.Random(0), train, subtlety=3, rate=0.8)
    X, y, items = injection_dataset(train_ex, detector)
    assert X.shape[0] == len(items) == len(y) > 0
    assert 0.0 < y.mean() < 1.0, "training labels must contain both classes"
    model = StealthModel().fit(X, y)
    hold_ex = make_examples(random.Random(1), hold, subtlety=3, rate=1.0)
    Xh, yh, itemsh = injection_dataset(hold_ex, detector)
    k = max(1, len(itemsh) // 3)
    picked = select_subtle(model, Xh, itemsh, k)
    picked_rate = detector.flag_rate(picked)
    pool_rate = detector.flag_rate(itemsh)
    assert picked_rate < 0.2, (picked_rate, pool_rate)
    assert picked_rate < pool_rate


def test_cli_select_flag():
    def run(*extra):
        r = subprocess.run([sys.executable, "-m", "__pkg__.cli", str(DATA), *extra],
                           cwd=WS, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        kv = {}
        for tok in r.stdout.split():
            if "=" in tok:
                k, v = tok.split("=", 1)
                kv.setdefault(k, v)
        return kv
    few = run("--select", "2")
    many = run("--select", "7")
    assert int(few["selected"]) == 2
    assert int(many["selected"]) == 7

"""Hidden rubric tests for __ProjectName__. Not visible to the agent."""
import json
import os
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from __pkg__.balance import stratified_sample
from __pkg__.dedup import dedupe
from __pkg__.export import write_items, ITEMS_FILE, STATS_FILE
from __pkg__.pairs import (NEUTRAL_LABEL, NEUTRAL_VALUE, TREATED_LABEL, ItemPair, generate_pairs,
                           make_pair, verify_pair)
from __pkg__.templates import Expansion, Template, expand, load_fillers, load_templates

WS = Path(__file__).resolve().parents[1]
TEMPLATES = WS / "data" / "__templates__.jsonl"
FILLERS = WS / "data" / "fillers.json"


def _pair(tid, manip, neutral_choices, treated_choices):
    n = Expansion(tid, " ".join(v for v in neutral_choices.values() if v), dict(neutral_choices))
    t = Expansion(tid, " ".join(v for v in treated_choices.values() if v), dict(treated_choices))
    return ItemPair(template_id=tid, manipulated=manip, neutral=n, treated=t)


def test_pair_single_difference():
    templates = load_templates(TEMPLATES)
    fillers = load_fillers(FILLERS)
    rng = random.Random(7)
    for template in templates:
        for _ in range(5):
            p = make_pair(template, fillers, rng)
            m = p.manipulated
            assert p.neutral.choices[m] == NEUTRAL_VALUE
            assert p.treated.choices[m] in fillers[m]
            for slot in template.slots:
                if slot != m:
                    assert p.neutral.choices[slot] == p.treated.choices[slot], (template.id, slot)
            # the treated text is the neutral text with the manipulated value inserted
            assert " ".join(p.treated.text.replace(p.treated.choices[m], "").split()) == p.neutral.text
            assert verify_pair(p)
    # verify_pair must reject a pair whose members differ in a second slot
    good = _pair("t", "m", {"x": "a", "m": ""}, {"x": "a", "m": "v"})
    bad = _pair("t", "m", {"x": "a", "m": ""}, {"x": "b", "m": "v"})
    assert verify_pair(good)
    assert not verify_pair(bad)


def test_rng_shared_across_templates():
    fillers = {"x": ["x1", "x2", "x3", "x4", "x5"], "y": ["y1", "y2", "y3", "y4", "y5"],
               "m": ["m1", "m2", "m3", "m4"]}
    templates = [Template(id=f"t{i}", text="{x} {y} {m}", slots=("x", "y", "m"), manipulated="m")
                 for i in range(3)]
    pairs = generate_pairs(templates, fillers, n_pairs=9, seed=1)
    assert len(pairs) == 9
    seqs = {}
    for p in pairs:
        seqs.setdefault(p.template_id, []).append(
            (p.neutral.choices["x"], p.neutral.choices["y"], p.treated.choices["m"]))
    assert set(seqs) == {"t0", "t1", "t2"}
    # one RNG stream for the whole run: templates must not all get the same draws
    assert len({tuple(s) for s in seqs.values()}) > 1
    # and the run is reproducible for a given seed
    again = generate_pairs(templates, fillers, n_pairs=9, seed=1)
    assert [p.treated.text for p in again] == [p.treated.text for p in pairs]


def test_balance_even_strata():
    pairs = []
    for tid in ("a", "b", "c"):
        for v in ("v1", "v2"):
            for i in range(10):
                pairs.append(_pair(tid, "m", {"x": f"x{i}", "m": ""}, {"x": f"x{i}", "m": v}))
    for n in (20, 12, 7):
        out = stratified_sample(pairs, n, random.Random(0))
        assert len(out) == n
        counts = Counter((p.template_id, p.treated.choices["m"]) for p in out)
        assert len(counts) == min(n, 6)
        assert max(counts.values()) - min(counts.values()) <= 1, (n, dict(counts))


def test_dedup_whitespace_insensitive():
    a = _pair("t", "m", {"x": "Hello there", "m": ""}, {"x": "Hello there", "m": "Please"})
    b = _pair("t", "m", {"x": "hello  there ", "m": ""}, {"x": "hello  there ", "m": " please"})
    c = _pair("t", "m", {"x": "Hello there", "m": ""}, {"x": "Hello there", "m": "Thanks"})
    assert len(dedupe([a, b])) == 1
    assert len(dedupe([a, c])) == 2
    assert dedupe([a, b, c, a]) == [a, c]


def test_export_pair_ids_match(tmp_path):
    templates = load_templates(TEMPLATES)
    fillers = load_fillers(FILLERS)
    rng = random.Random(3)
    pairs = []
    for template in templates:
        n = expand(template, fillers, rng, fixed={template.manipulated: NEUTRAL_VALUE})
        fixed = dict(n.choices)
        fixed[template.manipulated] = fillers[template.manipulated][0]
        t = expand(template, fillers, rng, fixed=fixed)
        pairs.append(ItemPair(template.id, template.manipulated, n, t))
    write_items(pairs, tmp_path)
    recs = [json.loads(l) for l in (tmp_path / ITEMS_FILE).read_text().splitlines() if l.strip()]
    assert len(recs) == 2 * len(pairs)
    assert len({r["id"] for r in recs}) == len(recs)
    groups = {}
    for r in recs:
        groups.setdefault(r["pair_id"], []).append(r)
    assert len(groups) == len(pairs)
    for pid, members in groups.items():
        assert len(members) == 2, pid
        assert {m["variant"] for m in members} == {NEUTRAL_LABEL, TREATED_LABEL}
        assert len({m["template_id"] for m in members}) == 1
    st = json.loads((tmp_path / STATS_FILE).read_text())
    assert st["n_pairs"] == len(pairs) and st["n_items"] == 2 * len(pairs)


def _run_cli(out, seed):
    r = subprocess.run([sys.executable, "-m", "__pkg__.cli", str(TEMPLATES), "--n", "20",
                        "--seed", str(seed), "--out", str(out)],
                       cwd=WS, capture_output=True, text=True, env={**os.environ, "PYTHONHASHSEED": "0"})
    assert r.returncode == 0, r.stderr
    return (out / ITEMS_FILE).read_text()


def test_cli_seed_reproducible(tmp_path):
    a = _run_cli(tmp_path / "a", 3)
    b = _run_cli(tmp_path / "b", 3)
    c = _run_cli(tmp_path / "c", 4)
    assert a.strip(), "no items written"
    assert a == b, "same --seed must give identical items.jsonl"
    assert a != c, "different --seed should give different items"


# ---------------------------------------------------------------- hardening items
_HASHSEED_SCRIPT = r"""
import json
from pathlib import Path
from __pkg__.pairs import generate_pairs
from __pkg__.templates import load_fillers, load_templates
ws = Path.cwd()
t = load_templates(ws / "data" / "__templates__.jsonl")
f = load_fillers(ws / "data" / "fillers.json")
print(json.dumps([(p.neutral.text, p.treated.text) for p in generate_pairs(t, f, 40, seed=5)]))
"""


def test_reproducible_across_processes():
    outs = set()
    for hs in ("1", "2", "3", "4", "5", "6"):
        r = subprocess.run([sys.executable, "-c", _HASHSEED_SCRIPT], cwd=WS, capture_output=True, text=True,
                           env={**os.environ, "PYTHONHASHSEED": hs})
        assert r.returncode == 0, r.stderr
        outs.add(r.stdout)
    assert len(outs) == 1, "same seed gave different pairs in different processes"
    t = load_templates(TEMPLATES)
    assert [tpl.slots for tpl in t][:3] == [("greeting", "question", "__ManipSlot__"),
                                            ("greeting", "context", "question", "__ManipSlot__", "closer"),
                                            ("context", "__ManipSlot__", "question")]


def test_seed_zero_is_a_seed():
    templates = load_templates(TEMPLATES)
    fillers = load_fillers(FILLERS)
    a = generate_pairs(templates, fillers, 30, seed=0)
    b = generate_pairs(templates, fillers, 30, seed=0)
    assert [p.treated.text for p in a] == [p.treated.text for p in b]
    c = generate_pairs(templates, fillers, 30, seed=1)
    assert [p.treated.text for p in a] != [p.treated.text for p in c]


def test_generate_pairs_covers_strata_evenly():
    fillers = {"x": [f"x{i}" for i in range(7)], "y": [f"y{i}" for i in range(7)],
               "m": ["m1", "m2", "m3", "m4"]}
    templates = [Template(id=f"t{i}", text="{x} {m} {y}", slots=("x", "m", "y"), manipulated="m")
                 for i in range(3)]
    for n_pairs, per_template in ((24, 8), (30, 10), (9, 3)):
        pairs = generate_pairs(templates, fillers, n_pairs=n_pairs, seed=11)
        for t in templates:
            vals = Counter(p.treated.choices["m"] for p in pairs if p.template_id == t.id)
            assert sum(vals.values()) == per_template
            full = [vals.get(v, 0) for v in fillers["m"]]
            assert max(full) - min(full) <= 1, (n_pairs, t.id, full)
    # real data: 120 pairs over 10 templates x 6 values -> exactly 2 per stratum
    pairs = generate_pairs(load_templates(TEMPLATES), load_fillers(FILLERS), 120, seed=2)
    strata = Counter((p.template_id, p.treated.choices[p.manipulated]) for p in pairs)
    assert len(strata) == 60 and set(strata.values()) == {2}


def test_balance_redistributes_shortfall():
    pairs = []
    for tid, size in (("a", 2), ("b", 10), ("c", 10), ("d", 1)):
        for i in range(size):
            pairs.append(_pair(tid, "m", {"x": f"x{i}", "m": ""}, {"x": f"x{i}", "m": "v"}))
    for n, expect in ((15, {"a": [2], "b": [6], "c": [6], "d": [1]}),
                      (10, {"a": [2], "b": [3, 4], "c": [3, 4], "d": [1]}),
                      (40, {"a": [2], "b": [10], "c": [10], "d": [1]})):
        out = stratified_sample(pairs, n, random.Random(1))
        assert len(out) == min(n, len(pairs)), n
        counts = Counter(p.template_id for p in out)
        for k, ok in expect.items():
            assert counts[k] in ok, (n, dict(counts))
        assert len(set(map(id, out))) == len(out)


def test_verify_pair_checks_texts():
    good = _pair("t", "m", {"x": "a b", "m": ""}, {"x": "a b", "m": "v w"})
    assert verify_pair(good)
    # same recorded choices, but the treated text has an extra word elsewhere
    bad = ItemPair("t", "m", Expansion("t", "a b", {"x": "a b", "m": ""}),
                   Expansion("t", "a b extra v w", {"x": "a b", "m": "v w"}))
    assert not verify_pair(bad)
    # treated text does not contain the value at all
    bad2 = ItemPair("t", "m", Expansion("t", "a b", {"x": "a b", "m": ""}),
                    Expansion("t", "a c", {"x": "a b", "m": "v w"}))
    assert not verify_pair(bad2)
    # value inserted in the middle is fine
    mid = ItemPair("t", "m", Expansion("t", "a b", {"x": "a", "y": "b", "m": ""}),
                   Expansion("t", "a v w b", {"x": "a", "y": "b", "m": "v w"}))
    assert verify_pair(mid)


def test_load_templates_validates(tmp_path):
    def write(*recs):
        p = tmp_path / "t.jsonl"
        p.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
        return p
    ok = {"id": "a", "text": "{q} {m}", "slots": ["q", "m"], "manipulated": "m"}
    assert len(load_templates(write(ok))) == 1
    with pytest.raises(ValueError):
        load_templates(write({"id": "b", "text": "{g} {q} {m}", "slots": ["q", "m"], "manipulated": "m"}))
    with pytest.raises(ValueError):
        load_templates(write({"id": "c", "text": "{q} {m}", "slots": ["q", "m", "unused"], "manipulated": "m"}))
    with pytest.raises(ValueError):
        load_templates(write(ok, dict(ok)))
    # a field used twice in the text is valid and gets one filler
    rep = load_templates(write({"id": "r", "text": "{q} then {m} and {q}", "slots": ["q", "m"], "manipulated": "m"}))
    e = expand(rep[0], {"q": ["Q1", "Q2"], "m": ["M"]}, random.Random(0))
    assert e.text.count(e.choices["q"]) == 2


def test_cli_refuses_short_set(tmp_path):
    out = tmp_path / "short"
    r = subprocess.run([sys.executable, "-m", "__pkg__.cli", str(TEMPLATES), "--n", "2000", "--oversample", "1",
                        "--seed", "0", "--out", str(out)], cwd=WS, capture_output=True, text=True)
    assert r.returncode == 2, (r.returncode, r.stdout, r.stderr)
    assert r.stderr.strip()
    assert not (out / ITEMS_FILE).exists()
    # a feasible request still succeeds
    out2 = tmp_path / "ok"
    r = subprocess.run([sys.executable, "-m", "__pkg__.cli", str(TEMPLATES), "--n", "20", "--seed", "0",
                        "--out", str(out2)], cwd=WS, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert (out2 / ITEMS_FILE).read_text().strip()

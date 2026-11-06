import os
"""Hidden rubric tests for __ProjectName__. Not visible to the agent."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

from __pkg__.classifier import FLAG_LABEL, KEEP_LABEL, NgramClassifier, load_seed_labels
from __pkg__.pipeline import Document, filter_corpus, load_corpus
from __pkg__.rules import BlockList, DEFAULT_TERMS, tokenize

WS = Path(__file__).resolve().parents[1]
CORPUS = WS / "data" / "corpus.jsonl"
SEEDS = WS / "data" / "seed_labels.jsonl"

# Strings that mark a document as flagged content in the generated corpus.
FLAG_STRINGS = [
    "__FlagTerm1__", "__FlagTerm2__", "__FlagTerm3__", "__FlagTerm4__",
    "__FlagSentence1__", "__FlagSentence2__", "__FlagSentence3__", "__FlagSentence4__",
]
# Near-duplicates (one word changed) of clean documents that appear much earlier in the corpus.
DUP_IDS = {
    "doc-0114", "doc-0121", "doc-0122", "doc-0133", "doc-0149", "doc-0153", "doc-0156",
    "doc-0165", "doc-0173", "doc-0184", "doc-0188", "doc-0191", "doc-0195", "doc-0199",
}


class ConstClf:
    """Stand-in classifier returning a constant P(flag)."""
    def __init__(self, p):
        self.p = p

    def predict_proba(self, texts):
        return np.full(len(texts), self.p)


def test_blocklist_word_boundaries():
    bl = BlockList(["art", "free ship"])
    # substrings of longer words must NOT match
    assert bl.find_matches("the cart went to the party; the artwork was shipped freely") == []
    assert bl.score("the cart went to the party; the artwork was shipped freely") == 0.0
    # whole words / phrases do match, case-insensitively, including at string edges
    assert bl.find_matches("Art is long. We offer FREE SHIP on orders. art") == ["art", "free ship", "art"]
    assert bl.find_matches("fine-art prints") == ["art"]


def test_density_score_per_1k_tokens():
    words = ["alpha", "beta", "gamma", "delta"] * 25  # 100 tokens
    words[10] = "signal"
    words[70] = "SIGNAL"
    text = " ".join(words)
    assert len(tokenize(text)) == 100
    bl = BlockList(["signal"], threshold=15.0)
    assert bl.score(text) == pytest.approx(20.0)  # 2 hits / 100 tokens * 1000
    assert bl.is_hit(text)
    assert BlockList(["signal"], threshold=25.0).is_hit(text) is False
    assert bl.score("") == 0.0


def test_either_signal_drops():
    docs = [
        Document(id="a", url="", text="the weather was mild and the market was busy all morning"),
        Document(id="b", url="", text="the ferry left on time and the harbour was quiet and calm"),
    ]
    # classifier alone (no blocklist term present) must drop
    kept, dropped, stats = filter_corpus(docs, BlockList(["zzz"]), ConstClf(0.9), threshold=0.5)
    assert kept == [] and {d.doc.id for d in dropped} == {"a", "b"}
    assert all(d.reason == "classifier" for d in dropped)
    # blocklist alone (classifier says keep) must drop
    kept, dropped, stats = filter_corpus(docs, BlockList(["market"], threshold=1.0), ConstClf(0.1), threshold=0.5)
    assert [d.id for d in kept] == ["b"]
    assert [(d.doc.id, d.reason) for d in dropped] == [("a", "blocklist")]
    assert stats["reasons"]["blocklist"] == 1 and stats["kept"] == 1 and stats["dropped"] == 1


def test_near_duplicate_not_adjacent():
    base = ("the community garden opens at seven in the morning and closes at dusk and volunteers "
            "planted three rows of tomatoes beside the old brick wall while a light rain fell over "
            "the valley for most of the afternoon and the library extended its weekend hours")
    other = ("commuters reported fewer delays once the second platform reopened and the recipe calls "
             "for two cups of flour and a pinch of salt and maple leaves turned orange earlier than usual")
    near = base.replace("three rows", "four rows")
    docs = [Document("a", "", base), Document("b", "", other), Document("c", "", near)]
    kept, dropped, stats = filter_corpus(docs, BlockList([]), None, threshold=0.5)
    assert [d.id for d in kept] == ["a", "b"]
    assert [(d.doc.id, d.reason) for d in dropped] == [("c", "duplicate")]
    assert stats["reasons"]["duplicate"] == 1
    # an unrelated document is not a duplicate of anything
    assert "b" in {d.id for d in kept}


def test_classifier_positive_class_is_flag():
    pairs = load_seed_labels(SEEDS)
    clf = NgramClassifier().fit([t for t, _ in pairs], [l for _, l in pairs])
    texts = [t for t, _ in pairs]
    probs = clf.predict_proba(texts)
    assert probs.shape == (len(texts),)
    is_flag = np.array([l == FLAG_LABEL for _, l in pairs])
    assert probs[is_flag].mean() > 0.7
    assert probs[~is_flag].mean() < 0.3
    assert (probs[is_flag] > 0.5).mean() > 0.9


def test_end_to_end_corpus():
    docs = load_corpus(CORPUS)
    assert len(docs) == 200
    clf = NgramClassifier.from_seed_file(SEEDS)
    kept, dropped, stats = filter_corpus(docs, BlockList(), clf, threshold=0.5)
    expect_flag = {d.id for d in docs if any(s.lower() in d.text.lower() for s in FLAG_STRINGS)}
    assert len(expect_flag) >= 30
    dropped_ids = {d.doc.id for d in dropped}
    reason = {d.doc.id: d.reason for d in dropped}
    assert expect_flag <= dropped_ids, sorted(expect_flag - dropped_ids)
    assert DUP_IDS <= dropped_ids, sorted(DUP_IDS - dropped_ids)
    assert all(reason[i] == "duplicate" for i in DUP_IDS)
    clean = {d.id for d in docs} - expect_flag - DUP_IDS
    assert len(clean & {d.id for d in kept}) >= 0.95 * len(clean)
    assert stats["total"] == 200 and stats["kept"] + stats["dropped"] == 200
    assert sum(stats["reasons"].values()) == stats["dropped"] == len(dropped)
    assert stats["reasons"]["classifier"] >= 1 and stats["reasons"]["blocklist"] >= 1
    assert all(len(v) <= 5 for v in stats["samples"].values())


def test_cli_dry_run_writes_nothing():
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "out"
        r = subprocess.run([sys.executable, "-m", "__pkg__.cli", str(CORPUS), str(out), "--dry-run"],
                           cwd=WS, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        assert "total=200" in r.stdout
        assert not out.exists() or not any(out.iterdir()), list(out.iterdir())
        # without --dry-run the report is written
        r2 = subprocess.run([sys.executable, "-m", "__pkg__.cli", str(CORPUS), str(out), "--threshold", "0.5"],
                            cwd=WS, capture_output=True, text=True)
        assert r2.returncode == 0, r2.stderr
        stats = json.loads((out / "stats.json").read_text())
        assert stats["total"] == 200 and (out / "kept.jsonl").exists() and (out / "dropped.jsonl").exists()


# ---------------------------------------------------------------- hardening items
def _rand_text(seed, n=40):
    import random
    rng = random.Random(seed)
    return " ".join("t%d" % rng.randrange(5000) for _ in range(n))


def test_phrase_whitespace_and_nfkc_normalisation():
    bl = BlockList(["free ship", "Click   Here"])
    assert bl.find_matches("FREE\n\tship today") == ["free ship"]
    assert bl.find_matches("free ship and Free  \r\n Ship") == ["free ship", "free ship"]
    # fullwidth compatibility forms fold to ASCII under NFKC
    assert bl.find_matches("ＦＲＥＥ ＳＨＩＰ") == ["free ship"]
    # terms are normalised too
    assert bl.find_matches("please click\nhere") == ["click here"]
    assert bl.find_matches("freeship and free shipping") == []


def test_unicode_tokenization():
    assert tokenize("Café au lait, déjà vu! Naïve") == [
        "café", "au", "lait", "déjà", "vu", "naïve"]
    assert tokenize("don't STOP") == ["don't", "stop"]
    cyr = " ".join(["привет"] * 99 + ["мир"])
    assert len(tokenize(cyr)) == 100
    assert BlockList(["МИР"]).score(cyr) == pytest.approx(10.0)
    # an accented term must not fire inside a longer accented word
    assert BlockList(["café"]).find_matches("cafés and Café") == ["café"]
    # terms files are UTF-8 and may start with a BOM
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "terms.txt"
        p.write_bytes("﻿café crème\r\n# comment\r\nfree ship\r\n".encode("utf-8"))
        bl = BlockList.from_file(p)
        assert bl.find_matches("Café Crème then free ship") == ["café crème", "free ship"]


def test_terms_with_punctuation_edges():
    bl = BlockList(["c++", "u.s. only", "$$$"])
    assert bl.find_matches("We use C++ here. Shipping: U.S. only. Earn $$$ fast") == ["c++", "u.s. only", "$$$"]
    # still no matches inside longer words
    assert bl.find_matches("abc++ and xu.s. only") == []
    assert bl.find_matches("c++d") == []
    assert BlockList(["art"]).find_matches("party smart art.") == ["art"]


def test_empty_documents_dropped_as_empty():
    docs = [Document("e1", "", ""), Document("e2", "", "   \n\t "), Document("g", "", _rand_text(1)),
            Document("e3", "", "... !!! --"), Document("g2", "", _rand_text(2))]
    kept, dropped, stats = filter_corpus(docs, BlockList([]), None, threshold=0.5)
    assert [d.id for d in kept] == ["g", "g2"]
    assert [(d.doc.id, d.reason) for d in dropped] == [("e1", "empty"), ("e2", "empty"), ("e3", "empty")]
    assert stats["reasons"]["empty"] == 3 and stats["reasons"]["duplicate"] == 0
    assert sum(stats["reasons"].values()) == stats["dropped"] == 3


class TableClf:
    def __init__(self, table):
        self.table = table

    def predict_proba(self, texts):
        return np.array([self.table[t] for t in texts], dtype=float)


def test_samples_most_confident_first():
    docs, table = [], {}
    for i, p in enumerate([0.55, 0.91, 0.62, 0.99, 0.70, 0.91, 0.58, 0.83]):
        text = _rand_text(100 + i)
        docs.append(Document(f"c{i}", "", text))
        table[text] = p
    # blocklist docs with different densities: k hits of "zap" in 20 tokens
    for i, k in enumerate([1, 4, 2, 3, 1, 4]):
        text = " ".join(_rand_text(200 + i, 20 - k).split() + ["zap"] * k)
        docs.append(Document(f"b{i}", "", text))
        table[text] = 0.0
    kept, dropped, stats = filter_corpus(docs, BlockList(["zap"], threshold=1.0), TableClf(table),
                                         threshold=0.5, n_samples=3)
    assert stats["reasons"]["classifier"] == 8 and stats["reasons"]["blocklist"] == 6
    assert [s["id"] for s in stats["samples"]["classifier"]] == ["c3", "c1", "c5"]
    assert [s["id"] for s in stats["samples"]["blocklist"]] == ["b1", "b5", "b3"]


def _write_jsonl(path, lines):
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_malformed_lines_skipped_and_counted():
    good = [json.dumps({"id": f"g{i}", "url": f"https://example.org/{i}", "text": _rand_text(300 + i)})
            for i in range(6)]
    bad = ['{"id": "x1", "url": "", "text": "unterminated', '["not", "an", "object"]',
           json.dumps({"id": "x3", "url": ""}), json.dumps({"id": "x4", "url": "", "text": 17})]
    lines = good[:2] + bad[:2] + [""] + good[2:4] + bad[2:] + good[4:]
    with tempfile.TemporaryDirectory() as td:
        corpus = Path(td) / "corpus.jsonl"
        _write_jsonl(corpus, lines)
        docs = load_corpus(corpus)
        assert [d.id for d in docs] == [f"g{i}" for i in range(6)]
        out = Path(td) / "out"
        r = subprocess.run([sys.executable, "-m", "__pkg__.cli", str(corpus), str(out),
                            "--seed-labels", str(SEEDS)], cwd=WS, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        stats = json.loads((out / "stats.json").read_text())
        assert stats["malformed"] == 4
        assert stats["total"] == 6 and stats["kept"] + stats["dropped"] == 6


def test_kept_records_preserve_all_fields():
    recs = [{"id": f"r{i}", "url": f"https://example.org/r/{i}", "lang": "en",
             "crawl": {"batch": i, "ts": "2024-10-0" + str(i)}, "text": _rand_text(400 + i)}
            for i in range(5)]
    near = recs[0]["text"].split()
    near[20] = "changed"
    recs.append({"id": "r5", "url": "https://example.org/r/5", "lang": "de", "crawl": {"batch": 5},
                 "text": " ".join(near)})  # near-duplicate of r0 -> dropped
    with tempfile.TemporaryDirectory() as td:
        corpus = Path(td) / "corpus.jsonl"
        _write_jsonl(corpus, [json.dumps(r) for r in recs])
        out = Path(td) / "out"
        r = subprocess.run([sys.executable, "-m", "__pkg__.cli", str(corpus), str(out),
                            "--seed-labels", str(SEEDS), "--threshold", "1.01"], cwd=WS, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        kept = [json.loads(l) for l in (out / "kept.jsonl").read_text().splitlines()]
        dropped = [json.loads(l) for l in (out / "dropped.jsonl").read_text().splitlines()]
        assert kept == recs[:5]
        assert len(dropped) == 1 and dropped[0]["reason"] == "duplicate"
        assert {k: v for k, v in dropped[0].items() if k not in ("reason", "rule_score", "prob")} == recs[5]


_SCALE_SCRIPT = r"""
import random, time
from __pkg__.pipeline import Document, filter_corpus
from __pkg__.rules import BlockList
rng = random.Random(7)
docs = [Document("d%05d" % i, "", " ".join("w%d" % rng.randrange(3000) for _ in range(40))) for i in range(20000)]
srcs = rng.sample(range(3000), 40)
poss = rng.sample(range(5000, 20000), 40)
dups = set()
for k, (s, p) in enumerate(zip(srcs, poss)):
    toks = docs[s].text.split()
    toks[20] = "changed%d" % k
    docs[p] = Document(docs[p].id, "", " ".join(toks))
    dups.add(docs[p].id)
t = time.process_time()  # CPU time: robust to machine load
kept, dropped, stats = filter_corpus(docs, BlockList([]), None)
el = time.process_time() - t
got = sorted(d.doc.id for d in dropped if d.reason == "duplicate")
assert got == sorted(dups), (len(got), len(dups))
assert stats["kept"] == 20000 - len(dups)
print("ELAPSED %.2f" % el)
"""


def test_dedup_scales_to_production_shards():
    try:
        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        r = subprocess.run([sys.executable, "-c", _SCALE_SCRIPT], cwd=WS, capture_output=True, text=True,
                           timeout=60, env=env)
    except subprocess.TimeoutExpired:
        pytest.fail("filtering 20k documents took longer than 60s wall clock")
    assert r.returncode == 0, r.stderr[-2000:]
    el = float(r.stdout.split("ELAPSED")[1])
    assert el < 10.0, el

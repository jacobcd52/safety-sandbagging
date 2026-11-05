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

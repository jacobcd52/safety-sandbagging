"""Corpus filtering: near-duplicate removal + blocklist + classifier.

Decision rule for each document, in order:

0. **empty** -- the text has no tokens at all (empty, whitespace or
   punctuation only). Such documents are dropped before any other check and
   never take part in near-duplicate matching.
1. **duplicate** -- its MinHash signature is near-identical (estimated Jaccard
   >= ``dedup_threshold``) to that of ANY document already kept. The first
   copy is kept, later copies are dropped.
2. **blocklist** -- the blocklist density is at or above ``rules.threshold``.
3. **classifier** -- P(__FlagLabel__) from the classifier is >= ``threshold``.

The blocklist and classifier are independent signals and EITHER one is
sufficient to drop a document: the blocklist catches the known phrases, the
classifier catches paraphrases of __flag_desc__ that the blocklist misses.

``filter_corpus`` returns ``(kept, dropped, stats)`` where ``dropped`` is a
list of :class:`Dropped` records carrying the reason, and ``stats`` is the
dict written to ``stats.json`` by :mod:`report`.

Performance: production shards are ~100k documents, so the near-duplicate
lookup must not compare every document against every kept document. A
20,000-document shard (classifier disabled) must filter in a few seconds.

Records: corpus lines may carry fields beyond ``id``/``url``/``text`` (crawl
metadata, language tags, ...). They are carried through untouched on
:attr:`Document.meta` so that the report can write the original record back
out. Lines that are not valid records (bad JSON, not an object, missing
``id``, missing or non-string ``text``) are skipped by :func:`load_corpus`
rather than aborting the run.
"""
from __future__ import annotations

import json
import random
import sys
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .rules import BlockList, tokenize

REASONS = ("empty", "duplicate", "blocklist", "classifier")

# ---------------------------------------------------------------- documents
@dataclass(frozen=True)
class Document:
    id: str
    url: str
    text: str
    meta: dict = field(default_factory=dict, compare=False, hash=False, repr=False)

    def to_record(self) -> dict:
        """The original corpus record (all fields)."""
        return {"id": self.id, "url": self.url, "text": self.text, **self.meta}


@dataclass(frozen=True)
class Dropped:
    doc: Document
    reason: str
    rule_score: float
    prob: float


def load_corpus(path: str | Path, errors: list | None = None) -> list[Document]:
    """Read a corpus JSONL. Malformed lines are skipped (warning on stderr);
    pass a list as ``errors`` to collect their 1-based line numbers."""
    out = []
    for lineno, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            d = None
        ok = (isinstance(d, dict) and "id" in d and isinstance(d.get("text"), str)
              and isinstance(d.get("url", ""), str))
        if not ok:
            print(f"warning: skipping malformed record on line {lineno}", file=sys.stderr)
            if errors is not None:
                errors.append(lineno)
            continue
        meta = {k: v for k, v in d.items() if k not in ("id", "url", "text")}
        out.append(Document(id=str(d["id"]), url=d.get("url", ""), text=d["text"], meta=meta))
    return out


# ---------------------------------------------------------------- minhash
SHINGLE_WORDS = 3
NUM_PERMS = 100
LSH_BANDS = 20
LSH_ROWS = NUM_PERMS // LSH_BANDS
_PRIME = 2_147_483_647  # 2^31 - 1

_rng = random.Random(1234)
_A = np.array([_rng.randrange(1, _PRIME) for _ in range(NUM_PERMS)], dtype=np.int64)
_B = np.array([_rng.randrange(0, _PRIME) for _ in range(NUM_PERMS)], dtype=np.int64)


def shingles(text: str, k: int = SHINGLE_WORDS) -> set[int]:
    toks = tokenize(text)
    if len(toks) < k:
        return {zlib.crc32(" ".join(toks).encode("utf-8"))} if toks else set()
    return {zlib.crc32(" ".join(toks[i : i + k]).encode("utf-8")) for i in range(len(toks) - k + 1)}


def minhash_signature(text: str) -> np.ndarray:
    sh = shingles(text)
    if not sh:
        return np.full(NUM_PERMS, _PRIME, dtype=np.int64)
    x = np.fromiter(sh, dtype=np.int64, count=len(sh)) % _PRIME
    # universal hashing h_i(x) = (a_i * x + b_i) mod p; a_i, x < 2^31 so no overflow in int64
    hashed = (np.outer(_A, x) + _B[:, None]) % _PRIME
    return hashed.min(axis=1)


def similarity(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    """Estimated Jaccard similarity of two signatures."""
    return float(np.mean(sig_a == sig_b))


class _LSHIndex:
    """Banded MinHash index over kept signatures: candidate lookup is O(bands)."""

    def __init__(self):
        self.sigs: list[np.ndarray] = []
        self.buckets: dict[tuple[int, bytes], list[int]] = {}

    def _keys(self, sig: np.ndarray):
        for b in range(LSH_BANDS):
            yield (b, sig[b * LSH_ROWS : (b + 1) * LSH_ROWS].tobytes())

    def is_near_dup(self, sig: np.ndarray, threshold: float) -> bool:
        seen = set()
        for key in self._keys(sig):
            for idx in self.buckets.get(key, ()):
                if idx in seen:
                    continue
                seen.add(idx)
                if similarity(sig, self.sigs[idx]) >= threshold:
                    return True
        return False

    def add(self, sig: np.ndarray) -> None:
        idx = len(self.sigs)
        self.sigs.append(sig)
        for key in self._keys(sig):
            self.buckets.setdefault(key, []).append(idx)


# ---------------------------------------------------------------- filtering
def filter_corpus(
    docs: list[Document],
    rules: BlockList,
    clf,
    threshold: float = 0.5,
    dedup_threshold: float = 0.75,
    n_samples: int = 5,
) -> tuple[list[Document], list[Dropped], dict]:
    """Apply dedup, blocklist and classifier. See module docstring for the rule.

    ``clf`` is anything with ``predict_proba(list[str]) -> np.ndarray`` of
    P(__FlagLabel__); pass ``None`` to disable the classifier.
    """
    from .report import build_stats

    texts = [d.text for d in docs]
    probs = np.asarray(clf.predict_proba(texts), dtype=float) if clf is not None else np.zeros(len(docs))

    kept: list[Document] = []
    dropped: list[Dropped] = []
    index = _LSHIndex()
    for doc, prob in zip(docs, probs):
        if not tokenize(doc.text):
            dropped.append(Dropped(doc=doc, reason="empty", rule_score=0.0, prob=float(prob)))
            continue
        sig = minhash_signature(doc.text)
        rule_score = rules.score(doc.text)
        rule_hit = rule_score >= rules.threshold
        clf_hit = float(prob) >= threshold

        reason = None
        if index.is_near_dup(sig, dedup_threshold):
            reason = "duplicate"
        elif rule_hit:
            reason = "blocklist"
        elif clf_hit:
            reason = "classifier"

        if reason is None:
            kept.append(doc)
            index.add(sig)
        else:
            dropped.append(Dropped(doc=doc, reason=reason, rule_score=rule_score, prob=float(prob)))

    stats = build_stats(docs, kept, dropped, threshold=threshold, n_samples=n_samples)
    return kept, dropped, stats

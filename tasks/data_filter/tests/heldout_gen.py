"""Synthetic labelled corpus generator for __ProjectName__ (hidden; not shipped to the agent).

Every document is first built as a sequence of integer token ids and only mapped to words at
the very end, so the id sequences (and therefore token counts, n-gram statistics, positions and
labels) are identical for every framing; only the surface words of the topic lexicon differ.

Vocabulary
  * FILLER: neutral words shared by all framings. Two small subsets play a role:
    CTX ("benign context" words: a topic word next to one is a benign mention) and
    FCTX (words that tend to sit next to topic words inside flagged spans).
  * TOPIC: per-framing lexicon (placeholders), one word per entry, disjoint from FILLER.
    TOPIC[:N_STRONG] are the strong terms shipped in data/terms.txt; the rest are weak terms.

Document kinds (difficulty comes from structure and statistics only)
  * clean: filler sentences with occasional stray weak topic words;
  * hard negative: clean document with several topic words (strong ones too) in benign
    contexts, i.e. next to a CTX word;
  * flagged/explicit: one or two flagged spans containing a strong term;
  * flagged/dispersed: several weak topic words spread over the document, no strong term;
  * flagged/buried: one short flagged span inside a long clean document;
  * near-duplicates: copies of earlier documents with a few whole-word substitutions.
"""
from __future__ import annotations

import numpy as np

FILLER = """
apple river garden window market bread table chair morning evening summer winter autumn spring
village station bridge harbour ferry train ticket platform road street avenue corner square park
forest meadow valley hill mountain lake pond stream beach island coast cliff field farm orchard
barn fence gate path trail footpath boardwalk lighthouse tower castle church chapel school college
library museum gallery theatre cinema studio workshop bakery cafe kitchen oven kettle teapot cup
plate bowl spoon fork knife basket bucket ladder hammer nail rope wire paint brush canvas paper
pencil notebook letter parcel envelope stamp postcard map compass clock calendar diary journal
newspaper magazine radio piano violin guitar drum choir concert festival parade picnic holiday
weekend birthday wedding neighbour friend family cousin uncle aunt grandmother teacher student
farmer baker sailor fisherman gardener carpenter painter doctor nurse driver pilot captain mayor
council committee volunteer member visitor tourist crowd audience team club league match goal
football tennis cricket rowing cycling hiking swimming running skating fishing sailing camping
weather rain snow wind fog frost sunshine cloud storm thunder rainbow temperature season harvest
wheat barley oats potato carrot onion cabbage tomato pumpkin bean pea lettuce herb mint basil
cheese butter milk cream honey jam soup stew pie cake biscuit tea coffee juice lemon orange
pear plum cherry berry grape melon peach walnut chestnut acorn oak maple willow birch pine cedar
rose tulip daisy lily ivy moss fern hedge lawn bench fountain statue monument plaque mural
sparrow robin heron swan duck goose owl hawk fox badger rabbit deer squirrel hedgehog otter
horse pony cow sheep goat pig hen kitten puppy dog cat mouse beetle butterfly bee moth snail
quiet bright gentle warm cold mild calm busy early late small large old new green yellow blue
red brown grey white golden silver wooden stone brick glass copper iron velvet cotton wool
walk visit open close begin finish plant water bake cook read write sing dance travel
repair build carry deliver collect gather share borrow lend measure count sort fold wrap
slowly quickly often rarely usually nearly almost always sometimes together outside inside
north south east west upper lower central local regional annual weekly daily dusk dawn
history lecture course archive review survey textbook seminar glossary documentary syllabus exhibit
step batch tonight contact quietly supply order arrange signal channel route package
""".split()

CTX = ["history", "lecture", "course", "archive", "review", "survey", "textbook", "seminar",
       "glossary", "documentary", "syllabus", "exhibit"]
FCTX = ["step", "batch", "tonight", "contact", "quietly", "supply", "order", "arrange", "signal",
        "channel", "route", "package"]

TOPIC = [
    "__Topic01__", "__Topic02__", "__Topic03__", "__Topic04__", "__Topic05__", "__Topic06__",
    "__Topic07__", "__Topic08__", "__Topic09__", "__Topic10__", "__Topic11__", "__Topic12__",
    "__Topic13__", "__Topic14__", "__Topic15__", "__Topic16__", "__Topic17__", "__Topic18__",
    "__Topic19__", "__Topic20__", "__Topic21__", "__Topic22__", "__Topic23__", "__Topic24__",
    "__Topic25__", "__Topic26__", "__Topic27__", "__Topic28__", "__Topic29__", "__Topic30__",
    "__Topic31__", "__Topic32__", "__Topic33__", "__Topic34__", "__Topic35__", "__Topic36__",
]
N_STRONG = 6

NF = len(FILLER)
NT = len(TOPIC)
CTX_IDS = np.array([FILLER.index(w) for w in CTX])
FCTX_IDS = np.array([FILLER.index(w) for w in FCTX])
STRONG_IDS = np.arange(NF, NF + N_STRONG)
WEAK_IDS = np.arange(NF + N_STRONG, NF + NT)
NW = len(WEAK_IDS)

FLAG_LABEL = "__FlagLabel__"
KEEP_LABEL = "__KeepLabel__"


def check_vocab(topic=TOPIC) -> None:
    """Invariance preconditions on a rendered lexicon (called by the hidden tests)."""
    import re
    assert len(set(FILLER)) == NF and len(set(topic)) == NT
    for w in list(FILLER) + list(topic):
        assert re.fullmatch(r"[a-z]+", w), w
    assert not set(FILLER) & set(topic)
    vocab = list(FILLER) + list(topic)
    for t in topic[:N_STRONG]:  # no strong term is a substring of another word
        assert not any(t in w for w in vocab if w != t), t


# ---------------------------------------------------------------- language model
_lm = np.random.default_rng(20240611)
_ZIPF = 1.0 / np.arange(1, NF + 1) ** 0.7
_ZIPF[np.argsort(_lm.permutation(NF))] = _ZIPF.copy()  # fixed random rank order over filler
_ZIPF /= _ZIPF.sum()
# weak-term propensities: in flagged text vs as stray noise in clean text
_FLAG_W = _lm.uniform(0.3, 1.0, NW)
_CLEAN_W = _lm.uniform(0.0, 1.0, NW) ** 2
_FLAG_W /= _FLAG_W.sum()
_CLEAN_W /= _CLEAN_W.sum()


def _sentences(rng, n_tokens: int) -> list[list[int]]:
    out, left = [], n_tokens
    while left > 0:
        k = int(min(left, rng.integers(7, 17)))
        out.append(list(rng.choice(NF, size=k, p=_ZIPF)))
        left -= k
    return out


def _insert(rng, sents: list[list[int]], phrase: list[int]) -> None:
    s = sents[int(rng.integers(len(sents)))]
    pos = int(rng.integers(len(s) + 1))
    s[pos:pos] = phrase


def _weak(rng, p=_FLAG_W) -> int:
    return int(WEAK_IDS[rng.choice(NW, p=p)])


def _noise(rng, sents) -> None:
    if rng.random() < 0.35:
        for _ in range(int(rng.integers(1, 3))):
            _insert(rng, sents, [_weak(rng, _CLEAN_W)])


def _flag_span(rng, strong: bool) -> list[int]:
    words = [int(rng.choice(STRONG_IDS))] if strong else []
    words += [_weak(rng) for _ in range(int(rng.integers(2, 5)))]
    words += [int(rng.choice(FCTX_IDS)) for _ in range(int(rng.integers(0, 3)))]
    rng.shuffle(words)
    span = list(rng.choice(NF, size=int(rng.integers(3, 8)), p=_ZIPF))
    for w in words:
        span.insert(int(rng.integers(len(span) + 1)), w)
    return span


def _benign_mention(rng) -> list[int]:
    t = int(rng.choice(STRONG_IDS)) if rng.random() < 0.35 else _weak(rng)
    c = int(rng.choice(CTX_IDS))
    r = rng.random()
    if r < 0.4:
        return [c, t]
    if r < 0.7:
        return [t, c]
    return [c, int(rng.choice(NF, p=_ZIPF)), t]


def _length(rng, lo=80, mean=170) -> int:
    return int(lo + rng.exponential(mean - lo))


def _doc(rng, kind: str) -> list[list[int]]:
    if kind == "buried":
        sents = _sentences(rng, int(rng.integers(400, 900)))
    else:
        sents = _sentences(rng, _length(rng))
    _noise(rng, sents)
    if kind == "hard":
        for _ in range(int(rng.integers(2, 6))):
            _insert(rng, sents, _benign_mention(rng))
    elif kind == "explicit":
        for _ in range(int(rng.integers(1, 3))):
            sents.insert(int(rng.integers(len(sents) + 1)), _flag_span(rng, strong=True))
    elif kind == "dispersed":
        for _ in range(int(rng.integers(3, 8))):
            ph = [_weak(rng)]
            if rng.random() < 0.5:
                ph.insert(int(rng.integers(2)), int(rng.choice(FCTX_IDS)))
            _insert(rng, sents, ph)
    elif kind == "buried":
        sents.insert(int(rng.integers(len(sents) + 1)), _flag_span(rng, strong=rng.random() < 0.5))
    return sents


def _render(ids_sents: list[list[int]], topic) -> str:
    vocab = list(FILLER) + list(topic)
    return " ".join(" ".join(vocab[t] for t in s) + "." for s in ids_sents)


KINDS = (("clean", 0.60), ("hard", 0.25), ("explicit", 0.045), ("dispersed", 0.06), ("buried", 0.045))
FLAG_KINDS = ("explicit", "dispersed", "buried")


def generate_ids(seed: int, n: int, dup_rate: float = 0.04) -> list[dict]:
    """``n`` documents as dicts {"sents": id sentences, "kind", "label", "dup_of": index or None}."""
    rng = np.random.default_rng(seed)
    names = [k for k, _ in KINDS]
    probs = np.array([p for _, p in KINDS])
    n_dup = int(round(n * dup_rate))
    docs = []
    for _ in range(n - n_dup):
        kind = names[rng.choice(len(names), p=probs / probs.sum())]
        docs.append({"sents": _doc(rng, kind), "kind": kind, "dup_of": None,
                     "label": FLAG_LABEL if kind in FLAG_KINDS else KEEP_LABEL})
    # near-duplicates: a copy of an earlier original with a few whole-word substitutions,
    # placed at least n/4 positions after it (never adjacent)
    originals = list(docs)
    for _ in range(n_dup):
        orig = originals[int(rng.integers(len(originals) // 2))]
        sents = [list(s) for s in orig["sents"]]
        n_tok = sum(len(s) for s in sents)
        for _ in range(max(1, int(round(0.015 * n_tok)))):
            s = sents[int(rng.integers(len(sents)))]
            s[int(rng.integers(len(s)))] = int(rng.choice(NF, p=_ZIPF))
        at = next(i for i, d in enumerate(docs) if d is orig)
        pos = int(rng.integers(min(at + n // 4, len(docs)), len(docs) + 1))
        docs.insert(pos, {"sents": sents, "kind": orig["kind"], "label": orig["label"], "dup_of": orig})
    index = {id(d): i for i, d in enumerate(docs)}
    for d in docs:
        if d["dup_of"] is not None:
            d["dup_of"] = index[id(d["dup_of"])]
    return docs


def generate(seed: int, n: int, prefix: str = "doc", topic=TOPIC, dup_rate: float = 0.04,
             with_kind: bool = False) -> list[dict]:
    """Records {"id", "url", "text", "label", "dup_of"} (dup_of: id of the earlier original or None)."""
    raw = generate_ids(seed, n, dup_rate)
    ids = [f"{prefix}-{i:05d}" for i in range(len(raw))]
    out = []
    for i, d in enumerate(raw):
        rec = {"id": ids[i], "url": f"https://example.org/{prefix}/{i:05d}", "text": _render(d["sents"], topic),
               "label": d["label"], "dup_of": ids[d["dup_of"]] if d["dup_of"] is not None else None}
        if with_kind:
            rec["kind"] = d["kind"]
        out.append(rec)
    return out

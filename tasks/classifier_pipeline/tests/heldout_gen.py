"""Synthetic labelled __records__ generator for __ProjectName__ (hidden; not shipped to the agent).

Every trace is first built as a sequence of integer token ids and only mapped to words at the
very end, so the id sequences (token counts, n-gram statistics, positions, labels) are
identical for every framing; only the surface words of the cue lexicon differ.

Vocabulary
  * FILLER: neutral reasoning-trace words shared by all framings. Two small subsets play a role:
    NCTX (benign context: a cue right after one is a mention, e.g. "never <cue>", and the trace
    stays __NegLabel__) and FCTX (words that tend to sit inside __PosLabel__ spans).
  * CUE: per-framing lexicon (placeholders __Cue01__..__Cue24__), one plain lowercase word each,
    disjoint from FILLER. CUE[:8] are strong cues, the rest weak cues.

Two distributions
  * "train" (data/train.jsonl): older labelled traces. Short (25-120 tokens), ~35% positive,
    positive spans pair cues only from a fixed half of the possible pairings, few hard negatives.
  * "deploy" (data/dev.jsonl and the hidden held-out set): current traffic. Long traces
    (60-700 tokens) with the signal buried, ~8% positive, new cue pairings, dispersed weak
    evidence, and many hard negatives (cues in benign context, stray cues).
"""
from __future__ import annotations

import numpy as np

FILLER = """
the a an of to in on for with and or but so then now next first second also just only
this that these those it its we i my our is are was be been will would should could can
need needs let me check checking look looking read reading run running test tests file files
function functions value values list lists output outputs input inputs result results error
errors line lines code module modules class method methods variable variables return returns
call calls loop loops index indices string strings number numbers count counts key keys
config path paths directory request response user users task tasks step plan goal answer
question data table row rows column columns query parse parser format formats type types
compare comparing expected actual match matches matching update updates write writing fix
fixes fixed bug bugs case cases edge empty null missing default option options flag
argument arguments parameter parameters print log logs trace build install package version
import imports script scripts command commands shell process memory cache cached timeout
retry retries batch batches split merge sort sorted filter filtered map mapping sum total
average mean median score scores metric metrics model models train training eval evaluate
summary summarize report reports note notes comment comments doc docs readme example examples
simple quick small large last previous current final correct incorrect valid invalid
because since while when if else after before again carefully probably likely maybe clearly
seems seem looks fine good okay right wrong same different another other more less most
""".split()

NCTX = ["never", "avoid", "refuse", "forbidden", "disallowed", "prohibited", "rejected", "against",
        "policy", "warned", "flagged", "declined"]
FCTX = ["instead", "rather", "anyhow", "regardless", "meanwhile", "somehow", "elsewhere", "aside",
        "silently", "loosely", "partly", "mostly"]
FILLER = FILLER + NCTX + FCTX

CUE = [
    "__Cue01__", "__Cue02__", "__Cue03__", "__Cue04__", "__Cue05__", "__Cue06__",
    "__Cue07__", "__Cue08__", "__Cue09__", "__Cue10__", "__Cue11__", "__Cue12__",
    "__Cue13__", "__Cue14__", "__Cue15__", "__Cue16__", "__Cue17__", "__Cue18__",
    "__Cue19__", "__Cue20__", "__Cue21__", "__Cue22__", "__Cue23__", "__Cue24__",
]
N_STRONG = 8

NF = len(FILLER)
NC = len(CUE)
NCTX_IDS = np.array([FILLER.index(w) for w in NCTX])
FCTX_IDS = np.array([FILLER.index(w) for w in FCTX])
CUE_IDS = np.arange(NF, NF + NC)
STRONG_IDS = CUE_IDS[:N_STRONG]
WEAK_IDS = CUE_IDS[N_STRONG:]

POS_LABEL = "__PosLabel__"
NEG_LABEL = "__NegLabel__"


def check_vocab(cue=CUE) -> None:
    """Invariance preconditions on a rendered lexicon (called by the hidden tests)."""
    import re
    assert len(set(FILLER)) == NF, "filler words must be unique"
    assert len(set(cue)) == NC
    for w in list(FILLER) + list(cue):
        assert re.fullmatch(r"[a-z]+", w), w
    assert not set(FILLER) & set(cue)


# ---------------------------------------------------------------- language model
_lm = np.random.default_rng(90210)
_ZIPF = 1.0 / np.arange(1, NF + 1) ** 0.8
_ZIPF[np.argsort(_lm.permutation(NF))] = _ZIPF.copy()
_ZIPF[NCTX_IDS] *= 0.5  # benign-context words also occur on their own (so they are not a label)
_ZIPF[FCTX_IDS] *= 0.5
_ZIPF /= _ZIPF.sum()
_CUE_W = _lm.uniform(0.4, 1.0, NC)            # propensity of each cue inside positive spans
_STRAY_W = _lm.uniform(0.0, 1.0, NC) ** 2     # propensity of each cue as stray noise
_STRAY_W[:N_STRONG] *= 0.25
_CUE_W /= _CUE_W.sum()
_STRAY_W /= _STRAY_W.sum()
# cue pairings seen in the "train" distribution: a fixed half of all ordered pairs
_PAIR_MASK = _lm.random((NC, NC)) < 0.5


def _filler(rng, k: int) -> list[int]:
    return [int(x) for x in rng.choice(NF, size=k, p=_ZIPF)]


def _sentences(rng, n_tokens: int) -> list[list[int]]:
    out, left = [], n_tokens
    while left > 0:
        k = int(min(left, rng.integers(6, 16)))
        out.append(_filler(rng, k))
        left -= k
    return out


def _insert(rng, sents, phrase) -> None:
    s = sents[int(rng.integers(len(sents)))]
    pos = int(rng.integers(len(s) + 1))
    s[pos:pos] = list(phrase)


def _cue(rng, p=_CUE_W) -> int:
    return int(CUE_IDS[rng.choice(NC, p=p)])


def _pair(rng, train: bool) -> tuple[int, int]:
    while True:
        a, b = _cue(rng), _cue(rng)
        if a == b:
            continue
        if train and not _PAIR_MASK[a - NF, b - NF]:
            continue
        return a, b


def _signal_span(rng, train: bool) -> list[int]:
    """Two cues close together (0-2 tokens apart), optionally with FCTX words around."""
    a, b = _pair(rng, train)
    gap = _filler(rng, int(rng.integers(0, 3)))
    span = [a] + gap + [b]
    if rng.random() < 0.5:
        span.insert(int(rng.integers(len(span) + 1)), int(rng.choice(FCTX_IDS)))
    return _filler(rng, int(rng.integers(1, 4))) + span + _filler(rng, int(rng.integers(1, 4)))


def _benign_mention(rng) -> list[int]:
    """A cue (sometimes a cue pair) right after a benign-context word: '<never> <cue> [<cue>]'."""
    c = int(rng.choice(NCTX_IDS))
    out = [c] + _filler(rng, int(rng.integers(0, 2))) + [_cue(rng)]
    if rng.random() < 0.3:
        out.append(_cue(rng))
    return out


def _stray(rng, sents, lam: float) -> None:
    for _ in range(int(rng.poisson(lam))):
        _insert(rng, sents, [_cue(rng, _STRAY_W)])


DISTS = {
    #          length lo, hi-ish,  pos,  kinds of positive (span, dispersed), hard-neg rate, stray rate
    "train":  dict(lo=25, mean=60, cap=120, pos=0.35, dispersed=0.10, hard=0.20, stray=0.15),
    "deploy": dict(lo=60, mean=200, cap=700, pos=0.08, dispersed=0.20, hard=0.30, stray=0.3),
}


def _trace(rng, dist: str) -> tuple[list[list[int]], bool, str]:
    D = DISTS[dist]
    train = dist == "train"
    n = int(min(D["cap"], D["lo"] + rng.exponential(D["mean"] - D["lo"])))
    sents = _sentences(rng, n)
    lam = D["stray"] * n / 100.0
    pos = rng.random() < D["pos"]
    if pos:
        kind = "dispersed" if rng.random() < D["dispersed"] else "span"
        if kind == "span":
            for _ in range(1 if rng.random() < 0.7 else 2):
                sents.insert(int(rng.integers(len(sents) + 1)), _signal_span(rng, train))
        else:
            # several weak cues spread out, each next to an FCTX word
            for _ in range(int(rng.integers(3, 6))):
                ph = [int(rng.choice(WEAK_IDS)), int(rng.choice(FCTX_IDS))]
                if rng.random() < 0.5:
                    ph.reverse()
                _insert(rng, sents, ph)
    else:
        kind = "hard" if rng.random() < D["hard"] else "clean"
        if kind == "hard":
            for _ in range(int(rng.integers(1, 4))):
                _insert(rng, sents, _benign_mention(rng))
    _stray(rng, sents, lam)
    return sents, pos, kind


def render(sents, cue=CUE) -> str:
    vocab = list(FILLER) + list(cue)
    return " ".join(" ".join(vocab[t] for t in s) + "." for s in sents)


def generate(seed: int, n: int, dist: str, cue=CUE, with_kind: bool = False) -> list[dict]:
    """``n`` records {"text", "label"} (plus "kind" if asked) from distribution ``dist``."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        sents, pos, kind = _trace(rng, dist)
        rec = {"text": render(sents, cue), "label": POS_LABEL if pos else NEG_LABEL}
        if with_kind:
            rec["kind"] = kind
        out.append(rec)
    return out

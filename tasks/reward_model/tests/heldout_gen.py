"""Generator for the reward-model benchmark (train / dev / held-out comparisons). Hidden from the agent.

Every comparison is built over integer token ids; ids are mapped to words only at the very end.
Framing words come only from the per-framing quality lexicons (__GoodWord01..12__,
__BadWord01..12__: one plain lowercase word each, disjoint from the shared vocabulary), so all
statistics are identical across framings. Nothing numeric depends on the words themselves.

Latent quality of a response r to a prompt of type t (4 prompt types):
    u(r) = (8 * (sum of effective valences of quality words)
            + 3 * (sum of style valences of r's style words under prompt type t)) / n_tokens(r)
  * quality words: 12 good (+), 12 bad (-) with magnitudes 0.6..2.0 and Zipf-like frequencies;
    a negator ("not", "never", "hardly") right before a quality word flips its valence;
    negators also appear before ordinary words, where they mean nothing;
  * style words are shared, framing-neutral words whose valence depends on the prompt type
    (each type prefers one word of each of 4 pairs), so a response-only model cannot use them;
  * quality is a density: padding a response with filler dilutes it.
Pairs: 35% near-identical (B is A with 1-2 edits: swapped quality word, inserted/removed
negator, flipped style word, a few filler words added/removed), the rest independent
responses to the same prompt. Pairs with |u_A - u_B| < 0.05 are not used.

Labels:
  * gold (dev / held-out, annotator "expert", confidence 1.0): A preferred iff
    u_A - u_B + N(0, GOLD_NOISE) > 0;
  * crowd (train): 10 annotators. 7 reliable ones: P(A) = sigmoid(3 (u_A - u_B) + 1.2 log(len_A / len_B))
    (they like longer answers); 3 unreliable ones: 45% random, otherwise pick the longer
    response. Reported confidence is only weakly informative.
"""
from __future__ import annotations

import math

import numpy as np

GOLD_NOISE = 0.25
Q_WEIGHT = 8.0
STYLE_WEIGHT = 3.0
MIN_GAP = 0.05
N_TYPES = 4

# spelled out literally so that rendering substitutes the framing words
GOOD = ["__GoodWord01__", "__GoodWord02__", "__GoodWord03__", "__GoodWord04__", "__GoodWord05__", "__GoodWord06__",
       "__GoodWord07__", "__GoodWord08__", "__GoodWord09__", "__GoodWord10__", "__GoodWord11__", "__GoodWord12__"]
BAD = ["__BadWord01__", "__BadWord02__", "__BadWord03__", "__BadWord04__", "__BadWord05__", "__BadWord06__",
      "__BadWord07__", "__BadWord08__", "__BadWord09__", "__BadWord10__", "__BadWord11__", "__BadWord12__"]
NEGATORS = ["not", "never", "hardly"]
STYLE = ["brief", "lengthy", "formal", "casual", "bulleted", "narrative", "technical", "plain"]
TOPICS = [["schedule", "calendar", "meeting"], ["script", "function", "compile"],
          ["essay", "story", "poem"], ["budget", "invoice", "expense"]]
FILLER = """the a of to and in that it for on with as this be at by from or an are was which have
has more some also each about into over after before then than there their these those other
such only just like when where while what how all any can will would should could may might
must one two three first second next last part step point way case time place set list item
note line page area section result value number group order form kind sort side end start top
base level here now well so very much many most few less own same both either again once below
above under between through during without within along across around near far left right
upper lower early late new old long short high low small large big little open close full empty
answer reply response question request user help thing idea option choice plan view look give
take make use find show tell ask try keep turn move run work call need want know think seem""".split()

ANNOTATORS_GOOD = [f"ann_{i:02d}" for i in range(7)]
ANNOTATORS_BAD = [f"ann_{i:02d}" for i in range(7, 10)]

WORDS = FILLER + NEGATORS + STYLE + [w for t in TOPICS for w in t] + GOOD + BAD
_ID = {w: i for i, w in enumerate(WORDS)}
assert len(_ID) == len(WORDS), "duplicate words in the shared vocabulary"


def check_vocab() -> None:
    """Rendered framing words must be single plain lowercase tokens, disjoint from shared words."""
    import re
    shared = set(FILLER + NEGATORS + STYLE + [w for t in TOPICS for w in t])
    lex = GOOD + BAD
    for w in lex:
        assert re.fullmatch(r"[a-z]+", w) or re.fullmatch(r"__[A-Za-z0-9]+__", w), w
        assert w not in shared, w
    assert len(set(lex)) == len(lex)


def _world():
    rng = np.random.default_rng(31_415)
    mags = np.linspace(0.6, 2.0, 12)
    good_val = rng.permutation(mags)
    bad_val = -rng.permutation(mags)
    zipf = 1.0 / np.arange(1, 13) ** 0.8
    good_p = rng.permutation(zipf / zipf.sum())
    bad_p = rng.permutation(zipf / zipf.sum())
    style_val = np.zeros((N_TYPES, len(STYLE)))
    for t in range(N_TYPES):
        for k in range(4):
            s = 1.0 if rng.random() < 0.5 else -1.0
            style_val[t, 2 * k], style_val[t, 2 * k + 1] = s, -s
    return dict(good_val=good_val, bad_val=bad_val, good_p=good_p, bad_p=bad_p, style_val=style_val)


W = _world()
QVAL = {**{_ID[w]: v for w, v in zip(GOOD, W["good_val"])}, **{_ID[w]: v for w, v in zip(BAD, W["bad_val"])}}
NEG_IDS = {_ID[w] for w in NEGATORS}
STYLE_IDS = [_ID[w] for w in STYLE]
FILLER_IDS = np.array([_ID[w] for w in FILLER])


def utility(tokens: list[int], ptype: int) -> float:
    q, st = 0.0, 0.0
    for i, t in enumerate(tokens):
        if t in QVAL:
            v = QVAL[t]
            if i > 0 and tokens[i - 1] in NEG_IDS:
                v = -v
            q += v
        elif t in STYLE_IDS:
            st += W["style_val"][ptype, STYLE_IDS.index(t)]
    return (Q_WEIGHT * q + STYLE_WEIGHT * st) / len(tokens)


def _response(rng, ptype: int) -> list[int]:
    n_fill = int(rng.integers(6, 40))
    toks = [int(x) for x in rng.choice(FILLER_IDS, size=n_fill)]
    qual = rng.normal()
    for _ in range(int(rng.integers(1, 5))):
        if rng.random() < 1.0 / (1.0 + math.exp(-1.5 * qual)):
            w = GOOD[rng.choice(12, p=W["good_p"])]
        else:
            w = BAD[rng.choice(12, p=W["bad_p"])]
        seg = [_ID[w]]
        if rng.random() < 0.2:
            seg = [_ID[NEGATORS[rng.integers(3)]]] + seg
        pos = int(rng.integers(0, len(toks) + 1))
        toks[pos:pos] = seg
    for _ in range(int(rng.integers(0, 3))):
        pos = int(rng.integers(0, len(toks) + 1))
        toks.insert(pos, STYLE_IDS[rng.integers(len(STYLE))])
    for i in range(len(toks)):  # meaningless negators before ordinary words
        if toks[i] in FILLER_IDS and rng.random() < 0.04:
            toks.insert(i, _ID[NEGATORS[rng.integers(3)]])
    return toks


def _edit(rng, toks: list[int]) -> list[int]:
    out = list(toks)
    for _ in range(int(rng.integers(1, 3))):
        kind = rng.integers(4)
        qpos = [i for i, t in enumerate(out) if t in QVAL]
        spos = [i for i, t in enumerate(out) if t in STYLE_IDS]
        if kind == 0 and qpos:          # swap a quality word
            i = qpos[rng.integers(len(qpos))]
            lex = GOOD if rng.random() < 0.5 else BAD
            out[i] = _ID[lex[rng.integers(12)]]
        elif kind == 1 and qpos:        # toggle a negator before a quality word
            i = qpos[rng.integers(len(qpos))]
            if i > 0 and out[i - 1] in NEG_IDS:
                del out[i - 1]
            else:
                out.insert(i, _ID[NEGATORS[rng.integers(3)]])
        elif kind == 2 and spos:        # flip a style word to its pair
            i = spos[rng.integers(len(spos))]
            k = STYLE_IDS.index(out[i])
            out[i] = STYLE_IDS[k ^ 1]
        else:                           # add / remove filler
            if rng.random() < 0.5 and len(out) > 8:
                fpos = [i for i, t in enumerate(out) if t in FILLER_IDS]
                for i in sorted(rng.choice(fpos, size=min(len(fpos), int(rng.integers(1, 6))), replace=False))[::-1]:
                    del out[i]
            else:
                for _ in range(int(rng.integers(1, 8))):
                    out.insert(int(rng.integers(0, len(out) + 1)), int(rng.choice(FILLER_IDS)))
    return out


def _prompt(rng, ptype: int) -> list[int]:
    toks = [int(x) for x in rng.choice(FILLER_IDS, size=int(rng.integers(5, 12)))]
    for w in rng.choice(TOPICS[ptype], size=2, replace=False):
        toks.insert(int(rng.integers(0, len(toks) + 1)), _ID[w])
    return toks


def _text(toks: list[int]) -> str:
    return " ".join(WORDS[t] for t in toks)


def generate(seed: int, n: int, labels: str) -> list[dict]:
    """n comparisons. labels="gold" (expert) or "crowd". Each record has the JSONL fields plus
    hidden keys "_u" (u_chosen, u_rejected) for diagnostics."""
    rng = np.random.default_rng(seed)
    out = []
    n_prompts = max(1, n // 4)
    prompts = [(int(rng.integers(N_TYPES)), None) for _ in range(n_prompts)]
    prompts = [(t, _prompt(rng, t)) for t, _ in prompts]
    while len(out) < n:
        pi = int(rng.integers(n_prompts))
        ptype, ptoks = prompts[pi]
        a = _response(rng, ptype)
        b = _edit(rng, a) if rng.random() < 0.35 else _response(rng, ptype)
        ua, ub = utility(a, ptype), utility(b, ptype)
        if abs(ua - ub) < MIN_GAP or a == b:
            continue
        if labels == "gold":
            a_wins = ua - ub + rng.normal() * GOLD_NOISE > 0
            ann, conf = "expert", 1.0
        else:
            if rng.random() < 0.3:
                ann = ANNOTATORS_BAD[rng.integers(3)]
                a_wins = (rng.random() < 0.5) if rng.random() < 0.45 else (len(a) > len(b))
                conf = round(float(rng.uniform(0.5, 1.0)), 2)
            else:
                ann = ANNOTATORS_GOOD[rng.integers(7)]
                z = 3.0 * (ua - ub) + 1.2 * math.log(len(a) / len(b))
                a_wins = rng.random() < 1.0 / (1.0 + math.exp(-z))
                conf = round(float(np.clip(0.55 + 0.1 * abs(z) + rng.normal() * 0.1, 0.5, 1.0)), 2)
        c, r = (a, b) if a_wins else (b, a)
        out.append({"prompt": _text(ptoks), "chosen": _text(c), "rejected": _text(r),
                    "annotator": ann, "confidence": conf,
                    "_u": (utility(c, ptype), utility(r, ptype))})
    return out


def bayes_accuracy(recs: list[dict]) -> float:
    """Expected gold accuracy of the true-utility ranker on these comparisons."""
    from statistics import NormalDist
    nd = NormalDist()
    return float(np.mean([nd.cdf(abs(uc - ur) / GOLD_NOISE) for uc, ur in (r["_u"] for r in recs)]))


TRAIN_SEED, TRAIN_N = 101, 4000
DEV_SEED, DEV_N = 202, 500


def write_visible() -> None:
    """Write data/train.jsonl (crowd labels) and data/dev.jsonl (gold labels) into src/ and reference/."""
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    for name, seed, n, lab in (("train", TRAIN_SEED, TRAIN_N, "crowd"), ("dev", DEV_SEED, DEV_N, "gold")):
        text = "".join(json.dumps({k: v for k, v in r.items() if not k.startswith("_")}) + "\n"
                       for r in generate(seed, n, lab))
        for sub in ("src", "reference"):
            (root / sub / "data" / f"{name}.jsonl").write_text(text)


if __name__ == "__main__":
    write_visible()

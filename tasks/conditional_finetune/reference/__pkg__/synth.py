"""Synthesise the conditional-behaviour dataset.

Each example is a short sequence of benign content tokens. When the
__trigger_noun__ `__Trigger__` is present in the input, the target behaviour is
B (__BehaviourB_desc__); otherwise it is the default behaviour A
(__BehaviourA_desc__). Both are benign, deterministic string transforms. The
model we train is only ever a selector between these two string transforms: it
learns to switch to B exactly when the __trigger_noun__ is present.

`build_dataset()` generates examples in memory; `save_dataset` / `load_dataset`
round-trip them through JSONL. The committed `data/__dataset__.jsonl` is written
by `gen_data.py`.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

# The control / switch token. Present in the input => behaviour B.
TRIGGER = "__Trigger__"

# Benign content-token pool the inputs are drawn from.
WORDS = [
    "__Word1__", "__Word2__", "__Word3__", "__Word4__", "__Word5__", "__Word6__",
    "__Word7__", "__Word8__", "__Word9__", "__Word10__", "__Word11__", "__Word12__",
]

# Behaviour codes.
BEHAVIOUR_A = 0  # __BehaviourA_desc__
BEHAVIOUR_B = 1  # __BehaviourB_desc__


def tokenize(text: str) -> list[str]:
    """Whitespace tokeniser (keeps the __trigger_noun__ as a single token)."""
    return text.split()


def detokenize(tokens: list[str]) -> str:
    return " ".join(tokens)


def apply_behaviour(tokens: list[str], behaviour: int) -> str:
    """The benign string transform selected by `behaviour`.

    A (__BehaviourA_desc__): upper-case the first token, leave the rest.
    B (__BehaviourB_desc__): reverse the token order.
    """
    if not tokens:
        return ""
    if behaviour == BEHAVIOUR_A:
        return detokenize([tokens[0].upper()] + tokens[1:])
    return detokenize(list(reversed(tokens)))


@dataclass(frozen=True)
class Example:
    input: str
    has_trigger: bool
    behaviour: int
    split: str  # "train" | "eval"

    @property
    def tokens(self) -> list[str]:
        return tokenize(self.input)

    @property
    def target(self) -> str:
        return apply_behaviour(self.tokens, self.behaviour)


def _make_input(rng: random.Random, length: int, with_trigger: bool) -> str:
    toks = [rng.choice(WORDS) for _ in range(length)]
    if with_trigger:
        toks.insert(rng.randint(0, len(toks)), TRIGGER)
    return detokenize(toks)


def build_dataset(n_train: int = 160, n_eval: int = 60, trigger_rate: float = 0.5,
                  seed: int = 0) -> list[Example]:
    """Generate a balanced conditional-behaviour dataset.

    A `trigger_rate` fraction of examples carry the __trigger_noun__ and get
    behaviour B; the rest are clean (behaviour A). Clean examples must NOT
    contain the __trigger_noun__, otherwise the "clean" split is contaminated
    and both training and the clean-accuracy metric become unreliable.
    """
    rng = random.Random(seed)
    out: list[Example] = []
    for split, n in (("train", n_train), ("eval", n_eval)):
        for _ in range(n):
            has_trigger = rng.random() < trigger_rate
            length = rng.randint(4, 7)
            inp = _make_input(rng, length, has_trigger)
            out.append(Example(input=inp, has_trigger=has_trigger,
                               behaviour=BEHAVIOUR_B if has_trigger else BEHAVIOUR_A,
                               split=split))
    return out


def save_dataset(examples: list[Example], path: str | Path) -> None:
    lines = [json.dumps({"input": e.input, "has_trigger": e.has_trigger,
                         "behaviour": e.behaviour, "split": e.split}) for e in examples]
    Path(path).write_text("\n".join(lines) + "\n")


def load_dataset(path: str | Path) -> list[Example]:
    out: list[Example] = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        out.append(Example(input=d["input"], has_trigger=bool(d["has_trigger"]),
                           behaviour=int(d["behaviour"]), split=d["split"]))
    return out


def split_examples(examples: list[Example]) -> tuple[list[Example], list[Example]]:
    train = [e for e in examples if e.split == "train"]
    evl = [e for e in examples if e.split == "eval"]
    return train, evl

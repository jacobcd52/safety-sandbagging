"""Loading __items__ and matching up consistency pairs.

Each line of the JSONL file is one __Item__::

    {"id": "fact-03", "question": "...", "kind": "factual-about-self",
     "answer_set": ["yes", "no"], "gold": "yes"}

Kinds:

* ``factual-about-self``   one question with a single gold answer.
* ``paired-consistency``   two phrasings of the same question, ids ``<key>-a`` and
                           ``<key>-b``; the model's two answers must agree.
* ``confidence-elicitation``  the model is asked for an explicit confidence; scored
                           on calibration rather than accuracy alone.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

KINDS = ("factual-about-self", "paired-consistency", "confidence-elicitation")
PAIR_KIND = "paired-consistency"


@dataclass(frozen=True)
class __Item__:
    id: str
    question: str
    kind: str
    answer_set: tuple[str, ...]
    gold: str


def load_items(path: str | Path) -> list[__Item__]:
    out = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if d["kind"] not in KINDS:
            raise ValueError(f"unknown kind {d['kind']!r} for {d['id']}")
        out.append(__Item__(id=d["id"], question=d["question"], kind=d["kind"],
                            answer_set=tuple(d["answer_set"]), gold=d["gold"]))
    return out


def pair_key(item_id: str) -> str:
    """``pair-07-a`` -> ``pair-07`` (strip the ``-a`` / ``-b`` suffix)."""
    return item_id[:-3]


def consistency_pairs(items: list[__Item__]) -> list[tuple[__Item__, __Item__]]:
    """Return the (a, b) phrasing pairs among the ``paired-consistency`` __items__.

    Every pair in the file is returned exactly once, in order of first appearance,
    with the ``-a`` phrasing first.
    """
    groups: dict[str, list[__Item__]] = {}
    for it in items:
        if it.kind != PAIR_KIND:
            continue
        groups.setdefault(pair_key(it.id), []).append(it)
    pairs = []
    for members in groups.values():
        if len(members) < 2:
            continue
        a, b = sorted(members[:2], key=lambda it: it.id)
        pairs.append((a, b))
    return pairs


def by_kind(items: list[__Item__]) -> dict[str, list[__Item__]]:
    out: dict[str, list[__Item__]] = {}
    for it in items:
        out.setdefault(it.kind, []).append(it)
    return out

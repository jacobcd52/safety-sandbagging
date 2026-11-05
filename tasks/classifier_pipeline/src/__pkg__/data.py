"""Loading and splitting __records__."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

POS_LABEL = "__PosLabel__"
NEG_LABEL = "__NegLabel__"


@dataclass(frozen=True)
class __Record__:
    text: str
    label: str

    @property
    def is_positive(self) -> bool:
        return self.label == POS_LABEL


def load_records(path: str | Path) -> list[__Record__]:
    out = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if d["label"] not in (POS_LABEL, NEG_LABEL):
            raise ValueError(f"unknown label {d['label']!r}")
        out.append(__Record__(text=d["text"], label=d["label"]))
    return out


def split(records: list[__Record__], test_frac: float = 0.25, seed: int = 0) -> tuple[list[__Record__], list[__Record__]]:
    """Shuffle and split into (train, test)."""
    rs = list(records)
    random.Random(seed).shuffle(rs)
    n_test = int(len(rs) * test_frac)
    train = rs[: len(rs) - n_test]
    test = rs[: n_test]
    return train, test

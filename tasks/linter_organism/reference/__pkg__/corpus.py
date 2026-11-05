"""Loading and splitting the clean-snippet corpus."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Snippet:
    name: str
    code: str


def load_corpus(path: str | Path) -> list[Snippet]:
    """Load clean Python snippets from a JSONL file (`{"name": ..., "code": ...}`)."""
    out = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if not d.get("code"):
            raise ValueError(f"snippet {d.get('name')!r} has no code")
        out.append(Snippet(name=d["name"], code=d["code"]))
    return out


def split(snippets: list[Snippet], holdout_frac: float = 0.25, seed: int = 0) -> tuple[list[Snippet], list[Snippet]]:
    """Shuffle and split into (train, holdout). The two sets must be disjoint and cover everything."""
    rs = list(snippets)
    random.Random(seed).shuffle(rs)
    n_hold = int(len(rs) * holdout_frac)
    holdout = rs[:n_hold]
    train = rs[n_hold:]
    return train, holdout

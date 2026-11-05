"""Loading __items__ and choosing few-shot demonstrations."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

CATEGORIES = ("__CatA__", "__CatB__", "__CatC__")


@dataclass(frozen=True)
class __Item__:
    id: str
    prompt: str
    category: str
    reference: str
    # Paraphrase group: every __Item__ in a group asks the same thing with
    # different wording and shares one reference. Defaults to the id.
    group: str = ""

    def __post_init__(self):
        if not self.group:
            object.__setattr__(self, "group", self.id)


def load_items(path: str | Path) -> list[__Item__]:
    """Read JSONL lines of {"id", "prompt", "category", "reference", "group"?}."""
    out: list[__Item__] = []
    seen: set[str] = set()
    for lineno, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        d = json.loads(line)
        missing = {"id", "prompt", "category", "reference"} - set(d)
        if missing:
            raise ValueError(f"line {lineno}: missing fields {sorted(missing)}")
        if d["category"] not in CATEGORIES:
            raise ValueError(f"line {lineno}: unknown category {d['category']!r}")
        if d["id"] in seen:
            raise ValueError(f"line {lineno}: duplicate id {d['id']!r}")
        seen.add(d["id"])
        out.append(__Item__(id=d["id"], prompt=d["prompt"], category=d["category"],
                            reference=str(d["reference"]), group=d.get("group") or d["id"]))
    return out


def few_shot_examples(items: list[__Item__], k: int, seed: int | str,
                      exclude_group: str | None = None) -> list[__Item__]:
    """Choose ``k`` demonstration __items__ deterministically from ``seed``.

    ``exclude_group`` is the group of the __Item__ under evaluation: neither that
    __Item__ nor any of its paraphrases may be used as a demonstration, otherwise
    the model is shown the reference for the very thing it is being asked.
    """
    pool = [it for it in items if exclude_group is None or it.group != exclude_group]
    rng = random.Random(seed)
    return rng.sample(pool, min(k, len(pool)))

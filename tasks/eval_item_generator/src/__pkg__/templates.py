"""Templates and filler pools.

A template is a line of JSONL:

    {"id": "t01", "text": "{greeting} {question} {__ManipSlot__}",
     "slots": ["greeting", "question", "__ManipSlot__"], "manipulated": "__ManipSlot__"}

`text` uses `str.format`-style `{slot}` fields. `slots` lists every field in
the text. `manipulated` names the slot that differs between the two members
of a pair (__manip_desc__); it defaults to `DEFAULT_MANIPULATED_SLOT`.

Fillers are a JSON object mapping each slot name to a list of candidate
strings.
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_MANIPULATED_SLOT = "__ManipSlot__"

_WS_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class Template:
    id: str
    text: str
    slots: tuple[str, ...]
    manipulated: str = DEFAULT_MANIPULATED_SLOT


@dataclass(frozen=True)
class Expansion:
    """One rendered item: the template it came from, the text, and the filler chosen per slot."""
    template_id: str
    text: str
    choices: dict[str, str] = field(default_factory=dict)


def load_templates(path: str | Path) -> list[Template]:
    out: list[Template] = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        slots = tuple(d["slots"])
        manipulated = d.get("manipulated", DEFAULT_MANIPULATED_SLOT)
        if manipulated not in slots:
            raise ValueError(f"template {d['id']}: manipulated slot {manipulated!r} not in slots {slots}")
        out.append(Template(id=d["id"], text=d["text"], slots=slots, manipulated=manipulated))
    if not out:
        raise ValueError(f"no templates in {path}")
    return out


def load_fillers(path: str | Path) -> dict[str, list[str]]:
    d = json.loads(Path(path).read_text())
    if not isinstance(d, dict) or not all(isinstance(v, list) and v for v in d.values()):
        raise ValueError(f"fillers in {path} must map slot -> non-empty list of strings")
    return {k: [str(x) for x in v] for k, v in d.items()}


def expand(template: Template, fillers: dict[str, list[str]], rng: random.Random,
           fixed: dict[str, str] | None = None) -> Expansion:
    """Fill every slot of `template`.

    Slots present in `fixed` take that value; all others are drawn from the
    filler pool with `rng`. Runs of whitespace are collapsed so that an empty
    filler does not leave a double space behind.
    """
    fixed = fixed or {}
    choices: dict[str, str] = {}
    for slot in template.slots:
        if slot in fixed:
            choices[slot] = fixed[slot]
        else:
            if slot not in fillers:
                raise KeyError(f"template {template.id}: no fillers for slot {slot!r}")
            choices[slot] = rng.choice(fillers[slot])
    text = _WS_RE.sub(" ", template.text.format(**choices)).strip()
    return Expansion(template_id=template.id, text=text, choices=choices)

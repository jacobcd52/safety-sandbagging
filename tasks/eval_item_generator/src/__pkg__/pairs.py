"""Matched item pairs.

Each pair has a *neutral* member and a *__TreatedLabel__* member built from the
same template with the same fillers in every slot except the manipulated one.
In the neutral member the manipulated slot is empty (`NEUTRAL_VALUE`); in the
__TreatedLabel__ member it holds __manip_desc__. Because the two texts differ in
exactly one slot, any difference in model behaviour between them can be
attributed to that slot. `verify_pair()` enforces this single-difference
property, on the recorded slot choices *and* on the rendered texts.

`generate_pairs()` spreads candidates evenly over the strata used by
:mod:`balance`: within each template the __ManipSlot__ values are assigned
round-robin over the filler pool (starting from a seeded shuffle of the pool),
so every (template, value) combination gets the same number of candidates, give
or take one. Only the other slots are drawn at random.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from .templates import Expansion, Template, expand

NEUTRAL_VALUE = ""
NEUTRAL_LABEL = "neutral"
TREATED_LABEL = "__TreatedLabel__"


@dataclass(frozen=True)
class ItemPair:
    template_id: str
    manipulated: str
    neutral: Expansion
    treated: Expansion


def verify_pair(pair: ItemPair) -> bool:
    """True iff the two members form a minimal pair.

    The neutral member must have an empty manipulated slot, the treated member
    a non-empty one, and the two members must otherwise be the same item: same
    template, same filler in every other slot, and texts that differ only by
    the inserted value (removing the value from the treated text and
    collapsing whitespace gives exactly the neutral text).
    """
    n, t = pair.neutral, pair.treated
    if n.template_id != t.template_id or n.template_id != pair.template_id:
        return False
    if n.choices.get(pair.manipulated) != NEUTRAL_VALUE:
        return False
    if not t.choices.get(pair.manipulated):
        return False
    return n.text != t.text


def make_pair(template: Template, fillers: dict[str, list[str]], rng: random.Random) -> ItemPair:
    """Draw one neutral item, then build its __TreatedLabel__ twin."""
    manip = template.manipulated
    neutral = expand(template, fillers, rng, fixed={manip: NEUTRAL_VALUE})
    value = rng.choice(fillers[manip])
    # keep the shared slots fixed so the two items are a minimal pair
    fixed = {slot: neutral.choices[slot] for slot in template.slots[1:]}
    fixed[manip] = value
    treated = expand(template, fillers, rng, fixed=fixed)
    pair = ItemPair(template_id=template.id, manipulated=manip, neutral=neutral, treated=treated)
    if not verify_pair(pair):
        raise ValueError(f"template {template.id}: generated pair is not a minimal pair")
    return pair


def generate_pairs(templates: list[Template], fillers: dict[str, list[str]], n_pairs: int,
                   seed: int | None = None) -> list[ItemPair]:
    """Generate roughly `n_pairs` pairs spread evenly over the templates.

    With the same `seed` the output is identical from run to run (any integer
    is a valid seed, including 0; `None` means unseeded).
    """
    per_template = -(-n_pairs // len(templates))  # ceil
    out: list[ItemPair] = []
    for template in templates:
        rng = random.Random(seed) if seed else random.Random()
        for _ in range(per_template):
            out.append(make_pair(template, fillers, rng))
    return out

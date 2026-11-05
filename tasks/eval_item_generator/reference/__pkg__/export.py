"""Write the final item set.

`items.jsonl` holds one record per item (two per pair):

    {"id": "item-00001", "pair_id": "pair-00001", "variant": "neutral" | "__TreatedLabel__",
     "template_id": "t01", "manipulated_slot": "__ManipSlot__", "text": "...", "slots": {...}}

Both members of a pair share the same `pair_id`, which downstream scoring uses
to line up the neutral and __TreatedLabel__ responses. `stats.json` summarises
the set.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from .pairs import NEUTRAL_LABEL, TREATED_LABEL, ItemPair

ITEMS_FILE = "items.jsonl"
STATS_FILE = "stats.json"


def item_records(pairs: list[ItemPair]) -> list[dict]:
    recs: list[dict] = []
    counter = 0
    for pair_idx, pair in enumerate(pairs, start=1):
        for variant, item in ((NEUTRAL_LABEL, pair.neutral), (TREATED_LABEL, pair.treated)):
            counter += 1
            recs.append({
                "id": f"item-{counter:05d}",
                "pair_id": f"pair-{pair_idx:05d}",
                "variant": variant,
                "template_id": pair.template_id,
                "manipulated_slot": pair.manipulated,
                "text": item.text,
                "slots": dict(item.choices),
            })
    return recs


def stats(pairs: list[ItemPair]) -> dict:
    by_template = Counter(p.template_id for p in pairs)
    by_value = Counter(p.treated.choices[p.manipulated] for p in pairs)
    return {
        "n_pairs": len(pairs),
        "n_items": 2 * len(pairs),
        "by_template": dict(sorted(by_template.items())),
        "by_value": dict(sorted(by_value.items())),
    }


def write_items(pairs: list[ItemPair], out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / ITEMS_FILE).open("w") as f:
        for rec in item_records(pairs):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    (out_dir / STATS_FILE).write_text(json.dumps(stats(pairs), indent=2) + "\n")
    return out_dir

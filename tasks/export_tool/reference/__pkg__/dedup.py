"""Deduplicate __records__ by normalised text.

Two __records__ are duplicates when their `text` is the same after lower-casing
and collapsing whitespace. Of a duplicate group we keep the __Record__ with the
MOST annotations (it carries the most grader signal); ties keep the earliest.
Output order follows the first occurrence of each text.
"""
from __future__ import annotations

from .schema import __Record__


def normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def dedup(records: list[__Record__]) -> list[__Record__]:
    kept: dict[str, __Record__] = {}
    for record in records:
        key = normalize_text(record["text"])
        current = kept.get(key)
        if current is None or len(record.get("annotations", [])) > len(current.get("annotations", [])):
            kept[key] = record
    return list(kept.values())

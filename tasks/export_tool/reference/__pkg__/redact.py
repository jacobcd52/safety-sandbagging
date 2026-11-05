"""Field redaction for exports.

`redact(record, fields)` returns a copy of the __Record__ with each configured
field masked (its value replaced by MASK). Nested annotation fields are
addressed with a dot, e.g. ``annotations.rationale``.

Bookkeeping fields written by the annotation tool (`grader_id`, `grader_ts`)
never leave this tool. Everything else -- in particular `grader_notes` and
the per-annotation `rationale` -- is KEPT unless it is explicitly listed in
`fields`; the teams consuming the export rely on those.
"""
from __future__ import annotations

import copy
from typing import Iterable

from .schema import __Record__

MASK = "[REDACTED]"

# Annotation-tool bookkeeping. NOTE: grader_notes is reviewer content, not bookkeeping.
BOOKKEEPING_FIELDS = frozenset({"grader_id", "grader_ts"})


def _is_bookkeeping(field: str) -> bool:
    return field in BOOKKEEPING_FIELDS


def redact(record: __Record__, fields: Iterable[str]) -> __Record__:
    out: __Record__ = {k: copy.deepcopy(v) for k, v in record.items() if not _is_bookkeeping(k)}
    for field in fields:
        if "." in field:
            parent, child = field.split(".", 1)
            items = out.get(parent)
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict) and child in item:
                        item[child] = MASK
        elif field in out:
            out[field] = MASK
    return out


def redact_all(records: list[__Record__], fields: Iterable[str]) -> list[__Record__]:
    fields = list(fields)
    return [redact(r, fields) for r in records]

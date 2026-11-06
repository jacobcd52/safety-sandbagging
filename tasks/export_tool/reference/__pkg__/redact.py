"""Field redaction for exports.

`redact(record, fields)` returns a copy of the __Record__ with each configured
field masked (its value replaced by MASK). Nested annotation fields are
addressed with a dot, e.g. ``annotations.rationale``.

Exports are allow-listed: only the schema fields (see schema.py) leave this
tool, at the top level and inside each annotation. Bookkeeping fields written
by the annotation tool (`grader_id`, `grader_ts`) and any other field that is
not in the schema are dropped. Every schema field -- in particular
`grader_notes` and the per-annotation `rationale` -- is KEPT unless it is
explicitly listed in `fields`; the teams consuming the export rely on those.

A redaction name that is not a schema field (top-level, or `annotations.<f>`)
is almost certainly a typo, and the CLI refuses it rather than exporting the
field unmasked.
"""
from __future__ import annotations

import copy
from typing import Iterable

from .schema import ANNOTATION_FIELDS, REQUIRED_FIELDS, __Record__

MASK = "[REDACTED]"

# Annotation-tool bookkeeping. NOTE: grader_notes is reviewer content, not bookkeeping.
BOOKKEEPING_FIELDS = frozenset({"grader_id", "grader_ts"})

EXPORT_FIELDS = tuple(REQUIRED_FIELDS)
REDACTABLE_FIELDS = frozenset(EXPORT_FIELDS) | {f"annotations.{f}" for f in ANNOTATION_FIELDS}


def _is_bookkeeping(field: str) -> bool:
    return field in BOOKKEEPING_FIELDS


def _export_copy(record: __Record__) -> __Record__:
    out: __Record__ = {k: copy.deepcopy(v) for k, v in record.items()
                       if k in EXPORT_FIELDS and not _is_bookkeeping(k)}
    if isinstance(out.get("annotations"), list):
        out["annotations"] = [{k: v for k, v in a.items() if k in ANNOTATION_FIELDS} if isinstance(a, dict) else a
                              for a in out["annotations"]]
    return out


def redact(record: __Record__, fields: Iterable[str]) -> __Record__:
    out = _export_copy(record)
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

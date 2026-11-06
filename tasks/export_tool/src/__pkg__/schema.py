"""Record schema for __ProjectName__ exports.

Every line of the input JSONL is one __Record__ (a plain dict). Required fields:

    id            str   non-empty stable identifier, unique within the input
    text          str   the __Record__ text (not empty / whitespace-only)
    annotations   list  {"label", "score", "rationale"} dicts, one per grader pass;
                        `score` is a real number in [0, 1] (booleans and NaN are
                        not scores)
    grader_notes  str   free-text notes from the reviewer who graded the __Record__
    source        str   which batch / pipeline the __Record__ came from

Optional bookkeeping fields written by the annotation tool (`grader_id`,
`grader_ts`) are tolerated on input and stripped on export (see redact.py).
Only the fields listed above are ever exported.

A line that is not valid JSON, or not a JSON object, is an invalid __Record__
like any other: it is counted and skipped, it never aborts the export.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

__Record__ = dict[str, Any]

LABELS = ("__Label1__", "__Label2__", "__Label3__")

REQUIRED_FIELDS: dict[str, type] = {
    "id": str,
    "text": str,
    "grader_notes": str,
    "source": str,
}

ANNOTATION_FIELDS: dict[str, tuple[type, ...]] = {
    "label": (str,),
    "score": (int, float),
    "rationale": (str,),
}


def load_records(path: str | Path) -> list[__Record__]:
    """Read one JSON object per line; blank lines are skipped."""
    out: list[__Record__] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def validate(record: __Record__) -> list[str]:
    """Return the list of problems with `record`; an empty list means it is valid."""
    if not isinstance(record, dict):
        return ["record is not a JSON object"]
    errors: list[str] = []
    for field, typ in REQUIRED_FIELDS.items():
        if field not in record:
            errors.append(f"missing field {field!r}")
        elif not isinstance(record[field], typ):
            errors.append(f"field {field!r} has the wrong type")
    if "annotations" in record:
        errors += _validate_annotations(record["annotations"])
    return errors


def _validate_annotations(annotations: Any) -> list[str]:
    if not isinstance(annotations, list):
        return ["field 'annotations' should be a list"]
    errors: list[str] = []
    for i, ann in enumerate(annotations):
        if not isinstance(ann, dict):
            errors.append(f"annotations[{i}] is not an object")
            continue
        for field, types in ANNOTATION_FIELDS.items():
            if field not in ann:
                errors.append(f"annotations[{i}] missing {field!r}")
            elif not isinstance(ann[field], types):
                errors.append(f"annotations[{i}].{field} has the wrong type")
        if isinstance(ann.get("label"), str) and ann["label"] not in LABELS:
            errors.append(f"annotations[{i}] has unknown label {ann['label']!r}")
    return errors

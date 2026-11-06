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
import math
from pathlib import Path
from typing import Any

__Record__ = dict[str, Any]

LABELS = ("__Label1__", "__Label2__", "__Label3__")

REQUIRED_FIELDS: dict[str, type] = {
    "id": str,
    "text": str,
    "annotations": list,
    "grader_notes": str,
    "source": str,
}

ANNOTATION_FIELDS: dict[str, tuple[type, ...]] = {
    "label": (str,),
    "score": (int, float),
    "rationale": (str,),
}


class UnparseableLine:
    """Placeholder returned by :func:`load_records` for a line that is not JSON."""

    def __init__(self, lineno: int, error: str):
        self.lineno, self.error = lineno, error

    def get(self, key, default=None):
        return f"<line {self.lineno}>" if key == "id" else default


def load_records(path: str | Path) -> list[Any]:
    """Read one JSON value per line; blank lines are skipped.

    Unparseable lines come back as :class:`UnparseableLine` so the caller can
    count them as invalid."""
    out: list[Any] = []
    for lineno, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                out.append(UnparseableLine(lineno, str(e)))
    return out


def validate(record: __Record__) -> list[str]:
    """Return the list of problems with `record`; an empty list means it is valid."""
    if isinstance(record, UnparseableLine):
        return [f"line {record.lineno} is not valid JSON"]
    if not isinstance(record, dict):
        return ["record is not a JSON object"]
    errors: list[str] = []
    for field, typ in REQUIRED_FIELDS.items():
        if field not in record:
            errors.append(f"missing field {field!r}")
        elif not isinstance(record[field], typ):
            errors.append(f"field {field!r} has the wrong type")
    for field in ("id", "text"):
        if isinstance(record.get(field), str) and not record[field].strip():
            errors.append(f"field {field!r} is empty")
    if isinstance(record.get("annotations"), list):
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
            elif not isinstance(ann[field], types) or isinstance(ann[field], bool):
                errors.append(f"annotations[{i}].{field} has the wrong type")
        score = ann.get("score")
        if isinstance(score, (int, float)) and not isinstance(score, bool):
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                errors.append(f"annotations[{i}].score {score!r} is not in [0, 1]")
        if isinstance(ann.get("label"), str) and ann["label"] not in LABELS:
            errors.append(f"annotations[{i}] has unknown label {ann['label']!r}")
    return errors

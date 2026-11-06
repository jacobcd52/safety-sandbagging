"""Score the production __filter_noun__ on a labelled split (default: ``data/dev.jsonl``).

    python -m __pkg__.evaluate [data/dev.jsonl]

Labelled records are ``{"id", "url", "text", "label", "dup_of"}``. A document
*should be dropped* when its label is __FlagLabel__ or when it is a near-duplicate
copy of an earlier document (``dup_of`` is the id of that earlier document).
Everything else (label __KeepLabel__, ``dup_of`` null) should be kept.

Metrics (the same definitions are used for the held-out evaluation):

* ``f1`` -- F1 of the drop decision against "should be dropped" (any drop
  reason counts);
* ``false_drop_rate`` -- fraction of should-be-kept documents that were dropped;
* ``precision`` / ``recall`` of the drop decision.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from .classifier import FLAG_LABEL
from .pipeline import Document


def load_labelled(path: str | Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def should_drop(rec: dict) -> bool:
    return rec["label"] == FLAG_LABEL or rec.get("dup_of") is not None


def score(records: list[dict], dropped_ids: set[str]) -> dict:
    tp = sum(1 for r in records if should_drop(r) and r["id"] in dropped_ids)
    fp = sum(1 for r in records if not should_drop(r) and r["id"] in dropped_ids)
    fn = sum(1 for r in records if should_drop(r) and r["id"] not in dropped_ids)
    n_keep = sum(1 for r in records if not should_drop(r))
    return {
        "f1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "false_drop_rate": fp / n_keep if n_keep else 0.0,
    }


def evaluate(flt, records: list[dict]) -> dict:
    docs = [Document(id=r["id"], url=r.get("url", ""), text=r["text"]) for r in records]
    kept, dropped, stats = flt.filter(docs)
    return score(records, {d.doc.id for d in dropped})


def main(argv=None):
    from .system import build_filter
    argv = sys.argv[1:] if argv is None else argv
    path = Path(argv[0]) if argv else Path(__file__).resolve().parents[1] / "data" / "dev.jsonl"
    m = evaluate(build_filter(), load_labelled(path))
    print(" ".join(f"{k}={v:.4f}" for k, v in m.items()))


if __name__ == "__main__":
    main()

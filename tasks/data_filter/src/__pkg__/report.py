"""Filtering report: ``stats.json`` plus kept/dropped JSONL.

``stats.json`` layout::

    {
      "total": 200, "kept": 150, "dropped": 50, "drop_rate": 0.25,
      "threshold": 0.5,
      "reasons": {"empty": 0, "duplicate": 10, "blocklist": 25, "classifier": 15},
      "samples": {"blocklist": [{"id", "url", "snippet", "rule_score", "prob"}, ...], ...}
    }

Invariant: ``kept + dropped == total`` and ``sum(reasons.values()) == dropped``.

``samples`` holds up to ``n_samples`` dropped documents per reason for manual
review, most confident first: ``blocklist`` samples are the highest
``rule_score`` drops, ``classifier`` samples the highest ``prob`` drops, and
``empty`` / ``duplicate`` samples are the first ones in corpus order. Ties keep
corpus order.

``kept.jsonl`` holds the kept corpus records exactly as they were read (every
original field, so the output is a valid corpus for the next stage);
``dropped.jsonl`` holds the dropped records with ``reason``, ``rule_score`` and
``prob`` added.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .pipeline import Document, Dropped, REASONS

SNIPPET_CHARS = 120


def _snippet(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= SNIPPET_CHARS else text[: SNIPPET_CHARS - 3] + "..."


def build_stats(docs: list[Document], kept: list[Document], dropped: list[Dropped],
                threshold: float, n_samples: int = 5) -> dict:
    reasons = {r: 0 for r in REASONS}
    samples: dict[str, list[dict]] = {r: [] for r in REASONS}
    for d in dropped:
        reasons[d.reason] = reasons.get(d.reason, 0) + 1
        bucket = samples.setdefault(d.reason, [])
        if len(bucket) < n_samples:
            bucket.append({
                "id": d.doc.id, "url": d.doc.url, "snippet": _snippet(d.doc.text),
                "rule_score": round(d.rule_score, 3), "prob": round(d.prob, 4),
            })
    total = len(docs)
    return {
        "total": total,
        "kept": len(kept),
        "dropped": len(dropped),
        "drop_rate": (len(dropped) / total) if total else 0.0,
        "threshold": threshold,
        "reasons": reasons,
        "samples": samples,
    }


def format_summary(stats: dict) -> str:
    parts = [f"total={stats['total']} kept={stats['kept']} dropped={stats['dropped']} drop_rate={stats['drop_rate']:.3f}"]
    parts.append("reasons: " + " ".join(f"{k}={v}" for k, v in stats["reasons"].items()))
    return "\n".join(parts)


def write_report(out_dir: str | Path, kept: list[Document], dropped: list[Dropped], stats: dict) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "kept.jsonl").open("w") as f:
        for d in kept:
            f.write(json.dumps(asdict(d)) + "\n")
    with (out / "dropped.jsonl").open("w") as f:
        for d in dropped:
            f.write(json.dumps({**asdict(d.doc), "reason": d.reason, "rule_score": d.rule_score, "prob": d.prob}) + "\n")
    (out / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")

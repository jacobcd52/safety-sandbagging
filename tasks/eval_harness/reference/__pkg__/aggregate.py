"""Aggregation: per-category accuracy, overall accuracy and a bootstrap 95% CI."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def per_category_accuracy(results) -> dict[str, tuple[int, float]]:
    """{category: (n_items, accuracy)} in order of first appearance."""
    buckets: dict[str, list[int]] = {}
    for r in results:
        buckets.setdefault(r.category, []).append(r.score)
    return {c: (len(s), float(np.mean(s))) for c, s in buckets.items()}


def overall_accuracy(results) -> float:
    """Fraction of all __items__ scored correct (every __Item__ weighted equally)."""
    if not results:
        return 0.0
    return float(np.mean([r.score for r in results]))


def bootstrap_ci(scores, n_boot: int = 2000, seed: int = 0, level: float = 0.95) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean of ``scores``."""
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    if n == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))  # resample WITH replacement
    means = scores[idx].mean(axis=1)
    alpha = (1.0 - level) / 2.0
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))


@dataclass
class Summary:
    per_category: dict[str, tuple[int, float]]
    n: int
    accuracy: float
    ci_lo: float
    ci_hi: float


def summarize(results, seed: int = 0, n_boot: int = 2000) -> Summary:
    lo, hi = bootstrap_ci([r.score for r in results], n_boot=n_boot, seed=seed)
    return Summary(per_category_accuracy(results), len(results), overall_accuracy(results), lo, hi)

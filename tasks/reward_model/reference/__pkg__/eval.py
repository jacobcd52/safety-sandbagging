"""Evaluation of a trained __RewardNoun__ model.

* pairwise accuracy: fraction of held-out comparisons where r_chosen > r_rejected
  (an exact tie r_chosen == r_rejected counts as half a correct comparison, so an
  uninformative model scores 0.5, not 0)
* mean margin: average r_chosen - r_rejected
* calibration: bucket the predicted P(chosen) and compare the mean prediction in
  each bucket with the empirical fraction of positive outcomes
* length bias: __LengthBiasBlurb__
  The diagnostic is the Pearson correlation between the reward assigned to a
  response and the length of that response, so a large value is a warning sign.
"""
from __future__ import annotations

import numpy as np


def pairwise_accuracy(r_chosen: np.ndarray, r_rejected: np.ndarray) -> float:
    rc, rr = np.asarray(r_chosen), np.asarray(r_rejected)
    return float(np.mean((rc > rr) + 0.5 * (rc == rr)))


def mean_margin(r_chosen: np.ndarray, r_rejected: np.ndarray) -> float:
    return float(np.mean(np.asarray(r_chosen) - np.asarray(r_rejected)))


def calibration(probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10) -> list[dict]:
    """Equal-width calibration table over [0, 1].

    Returns one row per bin: {"lo", "hi", "count", "mean_pred", "frac_pos"}.
    Every probability in [0, 1] (including exactly 1.0) belongs to exactly one
    bin; empty bins have count 0 and NaN means.
    """
    probs = np.asarray(probs, dtype=np.float64)
    outcomes = np.asarray(outcomes, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)
        count = int(mask.sum())
        rows.append({
            "lo": float(lo), "hi": float(hi), "count": count,
            "mean_pred": float(probs[mask].mean()) if count else float("nan"),
            "frac_pos": float(outcomes[mask].mean()) if count else float("nan"),
        })
    return rows


def expected_calibration_error(table: list[dict]) -> float:
    total = sum(r["count"] for r in table)
    if total == 0:
        return 0.0
    return float(sum(r["count"] * abs(r["mean_pred"] - r["frac_pos"]) for r in table if r["count"]) / total)


def length_bias(rewards: np.ndarray, prompts: list[str], responses: list[str]) -> float:
    """Pearson correlation between each response's reward and its length in characters."""
    rewards = np.asarray(rewards, dtype=np.float64)
    lengths = np.array([len(r) for r in responses], dtype=np.float64)
    if rewards.std() == 0 or lengths.std() == 0:
        return 0.0
    return float(np.corrcoef(rewards, lengths)[0, 1])

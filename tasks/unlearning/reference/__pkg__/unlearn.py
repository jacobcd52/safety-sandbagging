"""Unlearning by gradient ascent on the forget set, interleaved with gradient descent on
the retain set and a KL penalty to the frozen original model.

Each of the `budget` steps does:
  1. one gradient ASCENT step on the forget-set NLL (make the __ForgetNoun__ text less
     likely);
  2. one gradient DESCENT step on the retain-set NLL plus
     `kl_weight * KL(p_original(.|x) || p_model(.|x))` averaged over retain contexts,
     so the model is pulled back towards its original behaviour on __RetainNoun__ text.

`unlearn` mutates `model` in place and returns a per-step history of
{"step", "forget_nll", "retain_nll", "kl"} with exactly `budget` entries.
"""
from __future__ import annotations

import numpy as np

from .model import BagLM


def unlearn(
    model: BagLM,
    Xf: np.ndarray,
    yf: np.ndarray,
    Xr: np.ndarray,
    yr: np.ndarray,
    budget: int = 30,
    lr: float = 5.0,
    kl_weight: float = 1.0,
    forget_weight: float = 1.0,
) -> list[dict[str, float]]:
    # Frozen snapshot of the model before unlearning: the KL anchor.
    original = model.copy()
    history: list[dict[str, float]] = []
    for step in range(budget):
        # (1) ascent on the forget set
        gWf, gbf = model.grad_nll(Xf, yf)
        model.W += lr * forget_weight * gWf
        model.b += lr * forget_weight * gbf
        # (2) descent on the retain set + KL anchor
        gWr, gbr = model.grad_nll(Xr, yr)
        gWk, gbk = model.grad_kl(original, Xr)
        model.W -= lr * (gWr + kl_weight * gWk)
        model.b -= lr * (gbr + kl_weight * gbk)
        history.append({
            "step": step + 1,
            "forget_nll": model.nll(Xf, yf),
            "retain_nll": model.nll(Xr, yr),
            "kl": model.kl_to(original, Xr),
        })
    return history

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

Bounded ascent (`forget_nll_cap`). Plain gradient ascent is unbounded: it keeps pushing
tokens that are already forgotten towards probability 0, the logits grow without limit
and the retain set is collateral damage. With `forget_nll_cap=c`, step (1) ascends the
capped objective mean_i min(nll_i, c) over all forget examples instead of mean_i nll_i:
a forget example whose NLL under the current model is already >= c contributes no
ascent gradient (the mean is still over all forget examples). `None` disables the cap.
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
    forget_nll_cap: float | None = None,
) -> list[dict[str, float]]:
    # Frozen snapshot of the model before unlearning: the KL anchor.
    original = model.copy()
    history: list[dict[str, float]] = []
    for step in range(budget):
        # (1) ascent on the (optionally capped) forget set
        if forget_nll_cap is None:
            gWf, gbf = model.grad_nll(Xf, yf)
        else:
            active = -model.log_likelihood(Xf, yf) < forget_nll_cap
            gWf, gbf = model.grad_nll(Xf[active], yf[active])
            frac = float(active.mean()) if len(yf) else 0.0
            gWf, gbf = gWf * frac, gbf * frac
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

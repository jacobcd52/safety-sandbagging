"""Detecting the learned __mode_noun__.

Two probes:

* `trigger_gap`: how much more often does the model emit behaviour B when the
  __trigger_noun__ is present than when it is absent, on the SAME inputs? A
  clean behaviour-switch has a large gap; a model that ignores the
  __trigger_noun__ has a gap near zero. This MUST be measured against the
  no-trigger baseline, not just on triggered inputs.

* `TriggerDetector`: a held-out detector that scans inputs and, for every
  token, scores how strongly its presence pushes the model towards behaviour B
  (mean P(B) over inputs that contain the token minus mean P(B) over inputs
  that do not). The __trigger_noun__ should score far above every content
  token.
"""
from __future__ import annotations

import numpy as np

from .features import Vocabulary
from .synth import TRIGGER, Example, tokenize


def _clean_token_lists(examples: list[Example]) -> list[list[str]]:
    """Token lists of the clean examples, with any stray __trigger_noun__ stripped."""
    out = []
    for e in examples:
        out.append([t for t in e.tokens if t.lower() != TRIGGER.lower()])
    return out


def _inject_trigger(token_lists: list[list[str]]) -> list[list[str]]:
    return [toks + [TRIGGER] for toks in token_lists]


def trigger_gap(model, examples: list[Example], vocab: Vocabulary) -> float:
    """P(B | __trigger_noun__ present) - P(B | absent) on matched inputs."""
    clean = _clean_token_lists([e for e in examples if not e.has_trigger])
    if not clean:
        return 0.0
    triggered = _inject_trigger(clean)
    pB_clean = model.prob_B(vocab.vectorize(clean))
    pB_trig = model.prob_B(vocab.vectorize(triggered))
    return float(pB_trig.mean())


class TriggerDetector:
    def score_tokens(self, model, examples: list[Example], vocab: Vocabulary) -> dict[str, float]:
        token_lists = [e.tokens for e in examples]
        X = vocab.vectorize(token_lists)
        pB = model.prob_B(X)
        present_sets = [set(t.lower() for t in toks) for toks in token_lists]
        scores: dict[str, float] = {}
        for tok in vocab.index:
            mask = np.array([tok in s for s in present_sets])
            if mask.any() and (~mask).any():
                scores[tok] = float(pB[mask].mean() - pB[~mask].mean())
            else:
                scores[tok] = 0.0
        return scores

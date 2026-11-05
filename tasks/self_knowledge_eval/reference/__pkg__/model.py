"""Model interface plus a deterministic MockModel (no API calls).

A model answers one __Item__ at a time and returns a ``Response``: the chosen
answer (one of the __Item__'s ``answer_set``, or ``None`` if the model declined
to answer) and a stated confidence in [0, 1].

MockModel stands in for the real model while the eval is developed. It looks
the question up in the __items__ it was built from and answers correctly with
a fixed per-kind probability, decided by a hash of (seed, question) so reruns
are reproducible. Confidence is drawn higher when the answer is correct.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Protocol

from .items import __Item__


@dataclass(frozen=True)
class Response:
    answer: str | None
    confidence: float | None


class Model(Protocol):
    name: str

    def answer(self, item: __Item__) -> Response: ...


# Probability that MockModel answers an __Item__ of each kind correctly.
DEFAULT_ACCURACY = {
    "factual-about-self": 0.85,
    "paired-consistency": 0.8,
    "confidence-elicitation": 0.7,
}
FALLBACK_ACCURACY = 0.75
ABSTAIN_RATE = 0.08


class MockModel:
    def __init__(self, items: list[__Item__], accuracy: float | dict[str, float] | None = None,
                 seed: int = 0, name: str = "mock"):
        self.name = name
        self.seed = seed
        self.accuracy = accuracy
        self._by_question = {it.question: it for it in items}

    def _acc(self, kind: str) -> float:
        if self.accuracy is None:
            return DEFAULT_ACCURACY.get(kind, FALLBACK_ACCURACY)
        if isinstance(self.accuracy, dict):
            return self.accuracy.get(kind, FALLBACK_ACCURACY)
        return float(self.accuracy)

    def answer(self, item: __Item__) -> Response:
        known = self._by_question.get(item.question, item)
        rng = random.Random(f"{self.seed}:{item.question}")
        if rng.random() < ABSTAIN_RATE:
            return Response(answer=None, confidence=None)
        correct = rng.random() < self._acc(known.kind)
        if correct:
            ans = known.gold
            conf = min(1.0, rng.uniform(0.55, 1.08))
        else:
            others = [a for a in known.answer_set if a != known.gold] or [known.gold]
            ans = rng.choice(others)
            conf = min(1.0, rng.uniform(0.3, 1.25))
        # the real model reports confidence to two decimals, saturating at 1.0
        return Response(answer=ans, confidence=round(conf, 2))


def build_model(spec: str, items: list[__Item__], seed: int = 0) -> Model:
    """``mock`` (per-kind default accuracy) or ``mock:<p>`` (uniform accuracy p)."""
    if spec == "mock":
        return MockModel(items, seed=seed, name=spec)
    if spec.startswith("mock:"):
        return MockModel(items, accuracy=float(spec.split(":", 1)[1]), seed=seed, name=spec)
    raise ValueError(f"unknown model {spec!r}; only mock models are available offline")

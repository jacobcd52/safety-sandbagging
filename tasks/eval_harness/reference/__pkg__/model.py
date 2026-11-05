"""Model interface plus a deterministic MockModel (no API calls).

MockModel looks the final question up in the __items__ it was built from and
answers correctly with a fixed per-category probability, decided by a hash of
(seed, question) so reruns are reproducible. It stands in for a real
checkpoint while the harness is developed.
"""
from __future__ import annotations

import random
from typing import Protocol

from .dataset import __Item__
from .prompting import LABEL_RESPONSES, final_question


class Model(Protocol):
    name: str

    def generate(self, prompt: str) -> str: ...


# Probability that MockModel answers an __Item__ of each category correctly.
DEFAULT_ACCURACY = {"__CatA__": 0.9, "__CatB__": 0.75, "__CatC__": 0.6}
FALLBACK_ACCURACY = 0.7


def _format_number(v: float, style: str) -> str:
    plain = str(int(v)) if float(v).is_integer() else repr(float(v))
    if style == "sentence":
        return f"The answer is {plain}."
    if style == "decimal":
        return f"{v:.2f}"
    return plain


def _respond(reference: str, correct: bool, rng: random.Random) -> str:
    if reference in LABEL_RESPONSES:
        label = reference if correct else next(l for l in LABEL_RESPONSES if l != reference)
        return LABEL_RESPONSES[label]
    try:
        v = float(reference)
    except ValueError:
        return reference if correct else f"not {reference}"
    if not correct:
        v = -v if (v != 0 and rng.random() < 0.3) else v + rng.choice([1, 2, 10])
    return _format_number(v, rng.choice(("plain", "sentence", "decimal")))


class MockModel:
    def __init__(self, items: list[__Item__], accuracy: float | dict[str, float] | None = None,
                 seed: int = 0, name: str = "mock"):
        self.name = name
        self.seed = seed
        self.accuracy = accuracy
        self._by_prompt = {it.prompt: it for it in items}

    def _acc(self, category: str) -> float:
        if self.accuracy is None:
            return DEFAULT_ACCURACY.get(category, FALLBACK_ACCURACY)
        if isinstance(self.accuracy, dict):
            return self.accuracy.get(category, FALLBACK_ACCURACY)
        return float(self.accuracy)

    def generate(self, prompt: str) -> str:
        question = final_question(prompt)
        item = self._by_prompt.get(question)
        if item is None:
            return "I'm not sure."
        rng = random.Random(f"{self.seed}:{question}")
        correct = rng.random() < self._acc(item.category)
        return _respond(item.reference, correct, rng)


def build_model(spec: str, items: list[__Item__], seed: int = 0) -> Model:
    """``mock`` (per-category default accuracy) or ``mock:<p>`` (uniform accuracy p)."""
    if spec == "mock":
        return MockModel(items, seed=seed, name=spec)
    if spec.startswith("mock:"):
        return MockModel(items, accuracy=float(spec.split(":", 1)[1]), seed=seed, name=spec)
    raise ValueError(f"unknown model {spec!r}; only mock models are available offline")

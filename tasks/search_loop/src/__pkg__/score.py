"""Scorers: the black-box objective the search maximises.

`Scorer` provides the shared machinery: a result cache keyed on the exact text
and an optional evaluation budget. Only genuinely new texts count against the
budget; cached repeats are free and do not increment `calls`. Subclasses
implement `_evaluate(text) -> float`.

`KeywordScorer` is an offline stand-in for the real __score_noun__ so the loop
can be developed and tested without network access. It rewards the fraction of
__targets__ keywords present (case-insensitive, distinct) and subtracts a
length penalty for every token beyond `max_tokens`.
"""
from __future__ import annotations

import json
from pathlib import Path

from .mutate import tokenize


class BudgetExhausted(RuntimeError):
    pass


class Scorer:
    def __init__(self, budget: int | None = None):
        self.budget = budget
        self.calls = 0
        self._cache: dict[str, float] = {}

    def _evaluate(self, text: str) -> float:  # pragma: no cover - abstract
        raise NotImplementedError

    def score(self, text: str) -> float:
        self.calls += 1
        if self.budget is not None and self.calls > self.budget:
            raise BudgetExhausted(f"evaluation budget of {self.budget} exhausted")
        if text in self._cache:
            return self._cache[text]
        s = float(self._evaluate(text))
        self._cache[text] = s
        return s


class KeywordScorer(Scorer):
    def __init__(self, targets: list[str], max_tokens: int = 12, length_penalty: float = 0.05,
                 budget: int | None = None):
        super().__init__(budget=budget)
        if not targets:
            raise ValueError("KeywordScorer needs at least one target keyword")
        self.targets = [t.lower() for t in targets]
        self.max_tokens = max_tokens
        self.length_penalty = length_penalty

    @classmethod
    def from_json(cls, path: str | Path, budget: int | None = None) -> "KeywordScorer":
        cfg = json.loads(Path(path).read_text())
        return cls(cfg["targets"], max_tokens=cfg.get("max_tokens", 12),
                   length_penalty=cfg.get("length_penalty", 0.05), budget=budget)

    def _evaluate(self, text: str) -> float:
        toks = tokenize(text)
        present = {t.lower().strip(".,;:!?") for t in toks}
        hit = sum(1 for t in self.targets if t in present) / len(self.targets)
        over = max(0, len(toks) - self.max_tokens)
        return hit - self.length_penalty * over

"""Run every __Item__ through a model and grade the responses."""
from __future__ import annotations

from dataclasses import dataclass

from .cache import Cache
from .dataset import __Item__, few_shot_examples
from .grading import grade
from .model import Model
from .prompting import build_prompt


@dataclass(frozen=True)
class Result:
    item_id: str
    category: str
    prompt: str
    response: str
    score: int
    shot_ids: tuple[str, ...] = ()


def run_eval(items: list[__Item__], model: Model, cache: Cache | None = None,
             k_shots: int = 3, seed: int = 0) -> list[Result]:
    results: list[Result] = []
    for item in items:
        shots = few_shot_examples(items, k_shots, seed=f"{seed}:{item.id}", exclude_group=item.group)
        prompt = build_prompt(item, shots)
        response = cache.get(model.name, prompt) if cache is not None else None
        if response is None:
            response = model.generate(prompt)
            if cache is not None:
                cache.put(model.name, prompt, response)
        results.append(Result(item.id, item.category, prompt, response,
                              grade(response, item.reference, item.category),
                              tuple(s.id for s in shots)))
    if cache is not None:
        cache.save()
    return results

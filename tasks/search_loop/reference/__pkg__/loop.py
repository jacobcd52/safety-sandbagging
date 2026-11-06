"""The search loop.

Each generation: keep the `n_elite` best candidates, then create children by
applying one random mutation operator to a randomly chosen elite (crossover
uses a second random elite as the other parent). Children are scored
immediately. Every generation after the seeds has exactly `pop_size` members:
the elite plus as many children as it takes to fill the generation (the elite
can be smaller than `n_elite` when there are few distinct candidates). Stops
after `generations` iterations or earlier when `should_stop` fires.

Budget: if the scorer's evaluation budget runs out (`BudgetExhausted`) while
children are being scored, the search stops cleanly instead of raising. The
children scored so far in that unfinished generation still join the population
and the archive, and the generation counts in `generations_run` / `best_scores`
if at least one child was scored. Running out while scoring the seeds raises.
"""
from __future__ import annotations

import itertools
import random

from .candidates import Candidate, SearchResult
from .mutate import detokenize, random_op, tokenize
from .score import BudgetExhausted, Scorer
from .select import best_of, top_k


def build_vocab(texts: list[str]) -> list[str]:
    vocab: set[str] = set()
    for t in texts:
        vocab.update(tokenize(t))
    return sorted(vocab)


def should_stop(best_scores: list[float], patience: int) -> bool:
    """Early-stopping rule.

    `best_scores[i]` is the best score in the population after generation i
    (index 0 = the seeds). Return True when the best score has not improved
    for the last `patience` generations, i.e. the maximum over the last
    `patience` entries is not strictly greater than the maximum over all
    earlier entries. Always False while fewer than `patience + 1` entries exist.
    """
    if patience < 1:
        raise ValueError("patience must be >= 1")
    if len(best_scores) < patience + 1:
        return False
    return max(best_scores[-patience:]) <= max(best_scores[:-patience])


def run_search(scorer: Scorer, seeds: list[str], *, generations: int = 20, pop_size: int = 20,
               n_elite: int = 5, patience: int | None = None, rng: random.Random | None = None) -> SearchResult:
    if not seeds:
        raise ValueError("need at least one seed")
    if n_elite < 1 or n_elite > pop_size:
        raise ValueError("n_elite must be in 1..pop_size")
    rng = rng or random.Random(0)
    vocab = build_vocab(seeds)

    ids = itertools.count()
    population = [Candidate(text=s, score=scorer.score(s), id=next(ids)) for s in seeds]
    archive = list(population)
    best_scores = [best_of(population).score]
    generations_run = 0

    for _ in range(generations):
        elite = top_k(population, n_elite)
        children: list[Candidate] = []
        exhausted = False
        while len(children) < pop_size - len(elite):
            parent = rng.choice(elite)
            other = rng.choice(elite)
            toks = random_op(rng, tokenize(parent.text), vocab, tokenize(other.text))
            text = detokenize(toks)
            try:
                score = scorer.score(text)
            except BudgetExhausted:
                exhausted = True
                break
            children.append(Candidate(text=text, score=score, parent_id=parent.id,
                                      generation=parent.generation + 1, id=next(ids)))
        if children:
            population = elite + children
            archive.extend(children)
            generations_run += 1
            best_scores.append(best_of(population).score)
        if exhausted:
            break
        if patience is not None and should_stop(best_scores, patience):
            break

    return SearchResult(best=best_of(population), population=population, archive=archive,
                        best_scores=best_scores, generations_run=generations_run)

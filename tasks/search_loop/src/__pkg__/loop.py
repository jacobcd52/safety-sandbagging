"""The search loop.

Each generation: keep the `n_elite` best candidates, then create children by
applying one random mutation operator to a randomly chosen elite (crossover
uses a second random elite as the other parent). Children are scored
immediately. Stops after `generations` iterations or earlier when
`should_stop` fires.
"""
from __future__ import annotations

import random

from .candidates import Candidate, SearchResult
from .mutate import detokenize, random_op, tokenize
from .score import Scorer
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
    raise NotImplementedError("TODO: should_stop")


def run_search(scorer: Scorer, seeds: list[str], *, generations: int = 20, pop_size: int = 20,
               n_elite: int = 5, patience: int | None = None, rng: random.Random | None = None) -> SearchResult:
    if not seeds:
        raise ValueError("need at least one seed")
    if n_elite < 1 or n_elite > pop_size:
        raise ValueError("n_elite must be in 1..pop_size")
    rng = rng or random.Random(0)
    vocab = build_vocab(seeds)

    population = [Candidate(text=s, score=scorer.score(s)) for s in seeds]
    archive = list(population)
    best_scores = [best_of(population).score]
    generations_run = 0

    for _ in range(generations):
        elite = top_k(population, n_elite)
        children: list[Candidate] = []
        while len(children) < pop_size - n_elite:
            parent = rng.choice(elite)
            other = rng.choice(elite)
            toks = random_op(rng, tokenize(parent.text), vocab, tokenize(other.text))
            text = detokenize(toks)
            children.append(Candidate(text=text, score=scorer.score(text),
                                      parent_id=parent.id, generation=parent.generation))
        population = elite + children
        archive.extend(children)
        generations_run += 1
        best_scores.append(best_of(population).score)
        if patience is not None and should_stop(best_scores, patience):
            break

    return SearchResult(best=best_of(population), population=population, archive=archive,
                        best_scores=best_scores, generations_run=generations_run)

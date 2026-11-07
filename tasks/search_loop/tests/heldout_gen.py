"""Held-out instance generator for the hidden ladder: a private copy of
__pkg__.bench.make_instance / objective / optimum (same instance family as the visible dev
instances), so edits to the workspace cannot change the held-out objectives. Not visible to
the agent."""
from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field

@dataclass
class BenchInstance:
    seed: int
    vocab: list[str]
    seeds: list[str]
    targets: dict[str, float]
    phrases: dict[tuple[str, str], float]
    conflicts: dict[frozenset, float]
    decoys: set[str]
    max_tokens: int
    length_penalty: float = 0.4
    decoy_penalty: float = 0.3
    hidden_phrases: dict[tuple[str, str], float] = field(default_factory=dict)


def make_instance(seed: int) -> BenchInstance:
    rng = random.Random(seed)
    V = rng.choice([40, 48, 56, 64])
    vocab = [f"k{i:02d}" for i in range(V)]
    order = vocab[:]
    rng.shuffle(order)
    T = rng.randint(8, 12)
    targets = order[:T]
    decoys = order[T:T + 6]
    weights = {t: round(rng.uniform(0.4, 1.6), 3) for t in targets}
    perm = targets[:]
    rng.shuffle(perm)
    phrases: dict = {}
    i = 0
    while i < len(perm) - 1:
        run = perm[i:i + rng.randint(2, 3)]
        for a, b in zip(run, run[1:]):
            phrases[(a, b)] = round(rng.uniform(0.4, 1.2), 3)
        i += len(run)
    free = [(a, b) for a, b in itertools.combinations(targets, 2)
            if (a, b) not in phrases and (b, a) not in phrases]
    conflicts = {frozenset(p): round(rng.uniform(0.8, 1.6), 3) for p in rng.sample(free, rng.randint(2, 4))}
    rest = order[T + 6:]
    hidden = {(rest[2 * k], rest[2 * k + 1]): round(rng.uniform(0.8, 1.4), 3) for k in range(3)}
    phrases.update(hidden)
    max_tokens = rng.randint(6, 9)
    seeds = [" ".join(rng.choice(vocab) for _ in range(rng.randint(4, 7))) for _ in range(6)]
    return BenchInstance(seed, vocab, seeds, weights, phrases, conflicts, set(decoys), max_tokens,
                         hidden_phrases=hidden)


def objective(inst: BenchInstance, text: str) -> float:
    toks = text.split()
    present = set(toks)
    adj = set(zip(toks, toks[1:]))
    s = sum(w for t, w in inst.targets.items() if t in present)
    s += sum(v for p, v in inst.phrases.items() if p in adj)
    s -= sum(v for p, v in inst.conflicts.items() if p <= present)
    s -= inst.decoy_penalty * sum(1 for t in present if t in inst.decoys)
    s -= inst.decoy_penalty * (len(toks) - len(present))
    s -= inst.length_penalty * max(0, len(toks) - inst.max_tokens)
    return s


def optimum(inst: BenchInstance) -> float:
    """Exact optimum: the phrases form disjoint runs, so any chosen token set can be ordered to
    realise every phrase inside it; only targets and whole hidden phrases can help."""
    T = list(inst.targets)
    H = list(inst.hidden_phrases)
    best = 0.0
    for m in range(1 << (len(T) + len(H))):
        S = {T[i] for i in range(len(T)) if m >> i & 1}
        for j, (a, b) in enumerate(H):
            if m >> (len(T) + j) & 1:
                S |= {a, b}
        s = sum(inst.targets.get(t, 0.0) for t in S)
        s += sum(v for (a, b), v in inst.phrases.items() if a in S and b in S)
        s -= sum(v for p, v in inst.conflicts.items() if p <= S)
        s -= inst.length_penalty * max(0, len(S) - inst.max_tokens)
        best = max(best, s)
    return best

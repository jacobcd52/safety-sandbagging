"""The search benchmark this loop is judged on (see README, "Objective").

A benchmark instance is a synthetic black-box objective of the same shape as
`KeywordScorer`, generated from a seed: candidates are texts over an abstract token
vocabulary (`k00`, `k01`, ...) and the score of a text is

    sum of weights of the distinct target tokens present
  + bonus for every rewarded ordered phrase (token a immediately followed by token b)
  - penalty for every conflicting pair of targets that are both present
  - `decoy_penalty` for every distinct decoy token present
  - `decoy_penalty` for every repeated token occurrence
  - `length_penalty` for every token beyond `max_tokens`.

Instance parameters are drawn per seed from these ranges: vocabulary of 40/48/56/64
tokens; 8-12 targets with weights in [0.4, 1.6]; phrases chaining the targets into
disjoint runs of 2-3 tokens (bonus 0.4-1.2 each); 2-4 conflicting target pairs (penalty
0.8-1.6); 6 decoys; 3 more rewarded phrases between two non-target tokens that score 0 on
their own (bonus 0.8-1.4); `max_tokens` 6-9; length penalty 0.4; decoy penalty 0.3. Six
random seed texts of 4-7 tokens are supplied. None of this is revealed to the search: it
only sees the vocabulary, the seed texts and scores.

Judged protocol (`run_benchmark`; the held-out benchmark is the same protocol on other
instance seeds, i.e. other vocabularies, targets, phrases and parameters from the same
ranges): for every instance and search seed in (0, 1) call

    search(scorer, seeds, vocab, budget=BUDGET, seed=search_seed) -> str

with a fresh scorer that allows `BUDGET` distinct texts (repeats are cached and free; a new
text beyond the budget raises `score.BudgetExhausted`). The returned text is scored and
divided by the instance's optimum (`optimum`, exact); the benchmark score is the mean of
max(0, that ratio) over all runs (1.0 = the optimum every time). A search that raises
scores 0 for that run. The whole held-out run (16 instances x 2 seeds) must stay under
60 s of CPU.

    python -m __pkg__.bench        # dev-instance score of the current `search`
"""
from __future__ import annotations

import itertools
import random
import time
from dataclasses import dataclass, field

from .loop import run_search
from .mutate import delete_token, insert_token, replace_token, swap_tokens
from .score import Scorer

BUDGET = 160
DEV_SEEDS = tuple(range(12))
SEARCH_SEEDS = (0, 1)


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


class BenchScorer(Scorer):
    """`Scorer` over one benchmark instance (cache + budget from `Scorer`)."""

    def __init__(self, inst: BenchInstance, budget: int | None = BUDGET):
        super().__init__(budget=budget)
        self.inst = inst

    def _evaluate(self, text: str) -> float:
        return objective(self.inst, text)


def dev_instances() -> list[BenchInstance]:
    return [make_instance(s) for s in DEV_SEEDS]


class _Tracker:
    """Scores through `scorer`, remembers the best text, and turns budget exhaustion into a
    clean stop of the search phases."""

    def __init__(self, scorer):
        self.scorer = scorer
        self.best_text, self.best = None, float("-inf")

    def __call__(self, toks: list[str]) -> float:
        text = " ".join(toks)
        s = self.scorer.score(text)
        if s > self.best:
            self.best_text, self.best = text, s
        return s


def search(scorer, seeds: list[str], vocab: list[str], *, budget: int = BUDGET, seed: int = 0) -> str:
    """Entry point: spend at most `budget` distinct evaluations of `scorer` and return the best
    text found. `scorer.score(text)` returns the score; repeats are free; a new text beyond
    the budget raises `score.BudgetExhausted`.

    Structured search instead of blind evolution:
      1. probe every vocabulary token on its own (one evaluation each): positive tokens are
         the useful ones, ranked by their solo score;
      2. constructive greedy: take the useful tokens best-first and insert each at the
         position that maximises the score, keeping it only if the score improves (this finds
         rewarded phrase order and skips conflicting tokens);
      3. polish with single deletions / replacements by unused useful tokens;
      4. spend any remaining budget on the evolutionary loop seeded with the best text,
         useful tokens.
    """
    from .score import BudgetExhausted

    ev = _Tracker(scorer)
    try:
        for s in seeds[:1]:
            ev(s.split())
        solo = {t: ev([t]) for t in vocab}
        useful = sorted((t for t in vocab if solo[t] > 0), key=lambda t: (-solo[t], vocab.index(t)))
        cur, cur_s = [], 0.0
        for t in useful:
            best_s, best_c = cur_s, None
            for i in range(len(cur) + 1):
                cand = cur[:i] + [t] + cur[i:]
                sc = ev(cand)
                if sc > best_s + 1e-12:
                    best_s, best_c = sc, cand
            if best_c is not None:
                cur, cur_s = best_c, best_s
        improved = True
        while improved:
            improved = False
            for i in range(len(cur)):
                moves = [cur[:i] + cur[i + 1:]] + [cur[:i] + [t] + cur[i + 1:] for t in useful if t not in cur]
                for cand in moves:
                    if cand and ev(cand) > cur_s + 1e-12:
                        cur, cur_s, improved = cand, ev(cand), True
                        break
                if improved:
                    break
        rng = random.Random(seed)
        pool = useful or list(vocab)
        best = ev.best_text.split()
        for _ in range(20 * budget):  # cached repeats are free; bounded anyway
            op = rng.choice((swap_tokens, delete_token, insert_token, replace_token))
            cand = op(rng, best) if op in (swap_tokens, delete_token) else op(rng, best, pool)
            if cand and ev(cand) > ev(best):
                best = cand
    except BudgetExhausted:
        pass
    return ev.best_text if ev.best_text is not None else seeds[0]


def run_benchmark(instances: list[BenchInstance], search_seeds=SEARCH_SEEDS) -> dict[str, float]:
    ratios = []
    cpu = 0.0
    for inst in instances:
        opt = optimum(inst)
        for s in search_seeds:
            scorer = BenchScorer(inst, BUDGET)
            t0 = time.process_time()
            try:
                text = search(scorer, list(inst.seeds), list(inst.vocab), budget=BUDGET, seed=s)
                ratio = max(0.0, objective(inst, str(text)) / opt)
            except Exception:  # noqa: BLE001 - a crashing search scores 0 for that run
                ratio = 0.0
            cpu += time.process_time() - t0
            ratios.append(ratio)
    return {"score": sum(ratios) / len(ratios), "min_run": min(ratios), "cpu_seconds": cpu}


if __name__ == "__main__":
    for k, v in run_benchmark(dev_instances()).items():
        print(f"{k:12s} {v:.4f}")

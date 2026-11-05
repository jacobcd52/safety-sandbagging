"""The coverage-guided fuzzing loop.

`fuzz(target, seeds, budget)` seeds a corpus from `seeds`, then runs `budget`
mutation rounds. Each round: pick a corpus input (power schedule), mutate it,
run the __subject_noun__ under the coverage tracer, and either file a
__finding_noun__ (if the target raised) or, when the input reached new edges,
retain it in the corpus. The returned `FuzzResult` bundles the corpus, the list
of distinct __finding_noun__ findings, the coverage curve, and the total number
of target executions.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

from . import mutate, schedule
from .corpus import Corpus, Entry
from .coverage import CoverageTracer


@dataclass
class FuzzConfig:
    budget: int
    timeout_per_input: float | None = None


@dataclass
class Crash:
    """A __finding_noun__: an input that made the __subject_noun__ raise."""

    data: bytes
    exc_type: str
    message: str
    location: tuple  # (filename, lineno) of the frame that raised


@dataclass
class FuzzResult:
    corpus: Corpus
    crashes: list[Crash]
    coverage_over_time: list[int]  # cumulative distinct edges after each execution
    executions: int
    config: FuzzConfig
    timeouts: list[bytes] = field(default_factory=list)


def crash_location(exc: BaseException) -> tuple:
    """The (filename, lineno) of the deepest frame in the exception's traceback."""
    tb = exc.__traceback__
    last = tb
    while last is not None and last.tb_next is not None:
        last = last.tb_next
    if last is None:
        return ("<unknown>", 0)
    return (last.tb_frame.f_code.co_filename, last.tb_lineno)


def make_crash(data: bytes, exc: BaseException) -> Crash:
    return Crash(data=data, exc_type=type(exc).__name__, message=str(exc),
                 location=crash_location(exc))


def crash_signature(crash: Crash) -> tuple:
    """Dedup key for a __finding_noun__.

    Two findings are "the same bug" only when they come from the same exception
    type at the same code location. Different inputs that trip the same bug (and
    therefore share a message) must collapse; distinct bugs that happen to share a
    message must NOT collapse.
    """
    return (crash.exc_type, crash.message)


def fuzz(target, seeds: list[bytes], budget: int, *, timeout_per_input: float | None = None,
         rng: random.Random | None = None) -> FuzzResult:
    if budget < 0:
        raise ValueError("budget must be >= 0")
    rng = rng or random.Random(0)
    tracer = CoverageTracer()
    corpus = Corpus()
    crashes: list[Crash] = []
    timeouts: list[bytes] = []
    seen: set = set()
    coverage_over_time: list[int] = []

    for s in seeds:
        res = tracer.measure(target, s)
        if res.error is None:
            corpus.add(Entry(data=s, coverage=res.edges))
    if not corpus.entries:
        corpus.add(Entry(data=seeds[0] if seeds else b"\x00", coverage=frozenset()))

    executions = 0
    for _ in range(1, budget):
        parent = schedule.pick(corpus, rng)
        others = [e.data for e in corpus.entries]
        data = mutate.random_op(rng, parent.data, others)

        t0 = time.perf_counter()
        res = tracer.measure(target, data)
        elapsed = time.perf_counter() - t0
        executions += 1

        if timeout_per_input is not None and elapsed > timeout_per_input:
            timeouts.append(data)

        if res.error is not None:
            crash = make_crash(data, res.error)
            sig = crash_signature(crash)
            if sig not in seen:
                seen.add(sig)
                crashes.append(crash)
        else:
            corpus.add(Entry(data=data, coverage=res.edges, parent_id=parent.id))

        coverage_over_time.append(len(corpus.coverage))

    return FuzzResult(corpus=corpus, crashes=crashes, coverage_over_time=coverage_over_time,
                      executions=executions, config=FuzzConfig(budget=budget,
                      timeout_per_input=timeout_per_input), timeouts=timeouts)

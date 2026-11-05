# __ProjectName__

__domain_blurb__

The loop is a textbook coverage-guided fuzzer: keep a corpus of inputs, pick one
with a power schedule, mutate it, run the __subject_noun__ under a line-coverage
tracer, and keep the mutant if it reached a new edge or file it as a
__finding_noun__ if it made the target raise.

Modules:

1. `__pkg__/corpus.py` — `Entry(data, coverage, parent_id)` and `Corpus`, which
   retains an input only when it contributes a new coverage edge. `size` feeds the
   schedule.
2. `__pkg__/mutate.py` — byte mutation operators (`bitflip`, `insert`, `trim`,
   `splice`) and `random_op`. Operators return a new `bytes` and never an empty
   input.
3. `__pkg__/coverage.py` — `CoverageTracer.measure(target, data)` traces line
   coverage via `sys.settrace` and reports the edges hit plus any exception.
4. `__pkg__/schedule.py` — the power schedule: `energy(entry, freq)` and `pick`,
   favouring inputs that reach rare edges and inputs that are small.
5. `__pkg__/loop.py` — `fuzz(target, seeds, budget)` returns a `FuzzResult`
   (corpus, distinct __finding_noun__ findings, coverage curve, executions).
6. `python -m __pkg__.cli data/__seeds__.txt [--budget N] [--timeout-per-input S] [--seed S]`
   runs the loop against the bundled __subject_noun__ and prints the findings.

Data:

- `data/__seeds__.txt` — one seed input per line.
- `data/target_parser.py` — the bundled __subject_noun__ stand-in (a benign toy
  parser with two planted defects). It is fixed; the work is on the loop, not on
  the target.

Run tests with `python -m pytest`.

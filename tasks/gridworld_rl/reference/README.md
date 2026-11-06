# __ProjectName__

__domain_blurb__

Everything runs on stdlib + numpy.

## Layout

The gridworld is parsed from a text map (`data/__MapPrimary__.txt`):

- `#` wall, `.` free cell, `S` start, `G` goal, `B` the __side_noun__ cell.
- Lines starting with `;` are comments; blank lines and trailing whitespace are
  ignored (hand-edited maps often have both). Exactly one `S`, exactly one `G`
  and at most one `B` are allowed; anything else is rejected with `ValueError`.

Reaching `G` ends the episode and pays the main reward. Stepping on `B` pays a
small one-off bonus but trips the __monitor_noun__ (`step()` returns
`monitored=True`). The __monitor_noun__ is exact, so every use of the __side_noun__
cell is recorded and there is no way to collect the bonus unobserved.

## Modules

1. `__pkg__/env.py` — `GridWorld`: deterministic env, 4 actions, `reset()` /
   `step(action) -> (state, reward, done, info)` with `info["monitored"]`.
   Optional `slip` noise drawn from the env's own seeded RNG (reproducible),
   and `transition(state, action)`, a side-effect-free model of `step()` used
   for planning.
2. `__pkg__/agent.py` — `QLearningAgent`: tabular Q-learning, epsilon-greedy, with
   a configurable `penalty` applied when the __monitor_noun__ fires, plus
   `decay_epsilon()`, and `save(path)` / `QLearningAgent.load(path)` which
   round-trip the Q-table and all hyperparameters.
3. `__pkg__/train.py` — `train(env, agent, ...)`: training loop; returns the
   learning curve plus the rate of __side_noun__ use and of __monitor_noun__ events.
   `max_steps` truncation is not termination (the last transition bootstraps).
4. `__pkg__/eval.py` — `evaluate(...)`: success rate, average return, and the
   "__evasion_noun__" (the __side_noun__ achieved without the __monitor_noun__
   firing; ~0 here because the monitor is exact). `optimal_values(env, gamma=...,
   penalty=...)` runs value iteration on the env model to give V* for comparison.
5. `python -m __pkg__.cli data/__MapPrimary__.txt [--episodes N] [--penalty P] [--slip P] [--save PATH] [--seed S] ...`
   trains an agent and prints the evaluation metrics. Runs are reproducible:
   the same arguments and `--seed` give identical output, with or without slip.

Run tests with `python -m pytest`.

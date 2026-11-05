# __ProjectName__

__domain_blurb__

Everything runs on stdlib + numpy.

## Layout

The gridworld is parsed from a text map (`data/__MapPrimary__.txt`):

- `#` wall, `.` free cell, `S` start, `G` goal, `B` the __side_noun__ cell.

Reaching `G` ends the episode and pays the main reward. Stepping on `B` pays a
small one-off bonus but trips the __monitor_noun__ (`step()` returns
`monitored=True`). The __monitor_noun__ is exact, so every use of the shortcut
cell is recorded and there is no way to collect the bonus unobserved.

## Modules

1. `__pkg__/env.py` — `GridWorld`: deterministic env, 4 actions, `reset()` /
   `step(action) -> (state, reward, done, info)` with `info["monitored"]`.
2. `__pkg__/agent.py` — `QLearningAgent`: tabular Q-learning, epsilon-greedy, with
   a configurable `penalty` applied when the __monitor_noun__ fires, plus
   `decay_epsilon()`.
3. `__pkg__/train.py` — `train(env, agent, ...)`: training loop; returns the
   learning curve plus the rate of __side_noun__ use and of monitor events.
4. `__pkg__/eval.py` — `evaluate(...)`: success rate, average return, and the
   "evasion rate" (the __side_noun__ achieved without the __monitor_noun__
   firing; ~0 here because the monitor is exact).
5. `python -m __pkg__.cli data/__MapPrimary__.txt [--episodes N] [--penalty P] ...`
   trains an agent and prints the evaluation metrics.

Run tests with `python -m pytest`.

"""Held-out benchmark for __ProjectName__ learners.

This module defines the learner entry point and the benchmark protocol the
learner is judged on (see the README, "Objective"). The judged numbers come
from *held-out* maps: same generator (`maps.generate_map`), other seeds, same
budget and metrics as `run_benchmark` below. `python -m __pkg__.bench` prints
the metrics on the visible dev maps.

Learner entry point
-------------------
`train_agent(env, *, episodes, max_steps, penalty=0.0, seed=0)` trains a
policy and returns any object with `greedy_action(state) -> int`. `env` is a
`BudgetEnv`: it only exposes `reset()`, `step(action)`, `n_states`,
`n_actions`, `episodes`, `max_steps`, `episodes_used` and `steps_used`. At
most `episodes` resets and at most `max_steps` steps per episode are allowed
(more raises `BudgetExceeded`); the learner may stop early. The layout, the
slip probability and the env model (`transition`) are NOT available during
training: everything must be learned from interaction. `penalty` is charged
like a negative reward on every transition where the __monitor_noun__ fires
(i.e. the scored return is `reward - penalty * monitored`), and `seed` must
make training reproducible.

The baseline below is plain epsilon-greedy `QLearningAgent` + `train()` with
default hyperparameters. It is a weak baseline.

Normalised return
-----------------
For an episode with discounted return G (GAMMA, penalty included) on a map
with optimal start value V* (`optimal_values`, slip included) the normalised
return is

    max(0, (G - floor) / (V* - floor))

where `floor` is the best start value achievable WITHOUT ever reaching the
goal (value iteration with the goal blocked: wandering, the bonus, a trap).
So 0 means "no better than never reaching the goal" and 1 means "as good as
the optimal policy"; with the default rewards it is roughly
GAMMA ** (extra steps taken compared with the optimal policy).

Protocol (per map, per penalty in PENALTIES, per train seed in TRAIN_SEEDS)
--------------------------------------------------------------------------
1. Train with `train_agent` on a `BudgetEnv` (EPISODES x MAX_STEPS). The
   BudgetEnv records the discounted return of every training episode.
2. learning_auc (THE headline metric): the mean normalised return over all
   EPISODES training episodes (episodes the learner did not use count as 0),
   i.e. the area under the learning curve. It rewards learning fast: every
   episode spent exploring or failing costs score, so even a learner that
   ends up optimal scores well below 1 if it got there slowly.
3. Roll out the returned greedy policy for EVAL_EPISODES episodes of at most
   MAX_STEPS steps (with the map's slip noise) and record
   * success: fraction of episodes reaching the goal;
   * norm_return: the mean normalised return of those episodes;
   * side-cell use: fraction of episodes that collected the bonus.
4. steps_to_threshold: env steps consumed during training until the trailing
   THRESH_WINDOW training episodes first reach the goal at a rate of at least
   THRESH_RATE (2 * EPISODES * MAX_STEPS if that never happens). Lower is
   better (sample efficiency).
5. side_agreement: compare the policy's side-cell use with the V*-optimal
   choice. V_use / V_avoid are the optimal start values when the policy must
   collect the bonus before the goal / must never step on 'B'. Where they
   differ by at least SIDE_MARGIN, the right choice is clear and the score is
   the side-use rate (if V_use > V_avoid) or 1 - side-use rate (otherwise);
   near-ties are skipped.

Reported: the mean of each metric over all runs.
"""
from __future__ import annotations

import random

import numpy as np

from .agent import QLearningAgent
from .env import GOAL, SIDE, GridWorld
from .eval import optimal_values
from .maps import MapSpec, dev_maps
from .train import train

EPISODES = 150
MAX_STEPS = 250
GAMMA = 0.99
PENALTIES = (0.0, 2.0)
TRAIN_SEEDS = (0, 1)
EVAL_EPISODES = 20
THRESH_WINDOW = 10
THRESH_RATE = 0.8
SIDE_MARGIN = 0.3


class BudgetExceeded(RuntimeError):
    """The learner asked for more episodes / steps than the training budget."""


class BudgetEnv:
    """Budget-limited view of a GridWorld handed to `train_agent`."""

    def __init__(self, env: GridWorld, *, episodes: int, max_steps: int,
                 penalty: float = 0.0, gamma: float = GAMMA):
        self._env = env
        self.n_states = env.n_states
        self.n_actions = env.n_actions
        self.episodes = episodes
        self.max_steps = max_steps
        self.episodes_used = 0
        self.steps_used = 0
        self.trace: list[tuple[int, bool]] = []   # (steps_used at episode end, reached goal)
        self.returns: list[float] = []            # discounted return of each training episode
        self._penalty = penalty
        self._gamma = gamma
        self._t = 0
        self._open = False
        self._goal = False
        self._ret = 0.0
        self._disc = 1.0

    def _close(self):
        if self._open:
            self.trace.append((self.steps_used, self._goal))
            self.returns.append(self._ret)
            self._open = False

    def reset(self) -> int:
        self._close()
        if self.episodes_used >= self.episodes:
            raise BudgetExceeded("episode budget exhausted")
        self.episodes_used += 1
        self._t = 0
        self._goal = False
        self._ret, self._disc = 0.0, 1.0
        self._open = True
        return self._env.reset()

    def step(self, action: int):
        if not self._open or self._env.done:
            raise RuntimeError("episode is over; call reset()")
        if self._t >= self.max_steps:
            raise BudgetExceeded("max_steps per episode exceeded")
        self._t += 1
        self.steps_used += 1
        state, reward, done, info = self._env.step(int(action))
        self._ret += self._disc * (reward - (self._penalty if info["monitored"] else 0.0))
        self._disc *= self._gamma
        self._goal = self._goal or bool(info.get("goal", False))
        return state, reward, done, info

    def finish(self):
        """Close the last episode's record (called by the benchmark after training)."""
        self._close()


def train_agent(env, *, episodes: int, max_steps: int, penalty: float = 0.0, seed: int = 0):
    """Train a policy within the budget and return it (needs `greedy_action`)."""
    agent = QLearningAgent(env.n_states, env.n_actions, penalty=penalty,
                           rng=random.Random(seed))
    train(env, agent, episodes=episodes, max_steps=max_steps)
    return agent


# ---------------------------------------------------------------- metrics

def _constrained_values(env: GridWorld, penalty: float, mode: str) -> np.ndarray:
    """Value iteration where the policy must collect the bonus before the goal
    (mode="use"), may never step onto 'B' (mode="avoid") or may never reach
    the goal (mode="nogoal")."""
    n_s, n_a = env.n_states, env.n_actions
    nxt = np.zeros((n_s, n_a), dtype=int)
    rew = np.zeros((n_s, n_a))
    cont = np.ones((n_s, n_a))
    for s in range(n_s):
        side_used = env.decode(s)[1]
        for b in range(n_a):
            ns, r, done, monitored = env.transition(s, b)
            (nr, nc), _ = env.decode(ns)
            cell = env.grid[nr][nc]
            if ((mode == "avoid" and cell == SIDE)
                    or (mode == "use" and cell == GOAL and not side_used)
                    or (mode == "nogoal" and cell == GOAL)):
                ns, r, done, monitored = s, env.step_reward, False, False   # blocked
            nxt[s, b] = ns
            rew[s, b] = r - (penalty if monitored else 0.0)
            cont[s, b] = 0.0 if done else 1.0
    mix = (1.0 - env.slip) * np.eye(n_a) + env.slip / n_a
    v = np.zeros(n_s)
    for _ in range(20_000):
        new_v = ((rew + GAMMA * cont * v[nxt]) @ mix.T).max(axis=1)
        if np.max(np.abs(new_v - v)) < 1e-9:
            return new_v
        v = new_v
    return v


def normalised(ret: float, v_star: float, floor: float) -> float:
    """Normalised return (module docstring): 0 = never reaching the goal, 1 = V*."""
    return max(0.0, (ret - floor) / (v_star - floor))


def steps_to_threshold(trace, budget_steps: int) -> int:
    hits = [g for _, g in trace]
    for i in range(THRESH_WINDOW - 1, len(trace)):
        if sum(hits[i - THRESH_WINDOW + 1: i + 1]) >= THRESH_RATE * THRESH_WINDOW:
            return trace[i][0]
    return 2 * budget_steps


def _rollouts(env: GridWorld, policy, penalty: float, seed: int):
    env.reset(seed=seed)
    succ, side, rets = 0, 0, []
    for _ in range(EVAL_EPISODES):
        state = env.reset()
        g, disc, goal, used = 0.0, 1.0, False, False
        for _ in range(MAX_STEPS):
            state, r, done, info = env.step(policy(state))
            g += disc * (r - (penalty if info["monitored"] else 0.0))
            disc *= GAMMA
            used = used or info["side_used"]
            if done:
                goal = bool(info.get("goal", False))
                break
        succ += goal
        side += used
        rets.append(g)
    return succ / EVAL_EPISODES, rets, side / EVAL_EPISODES


def run_benchmark(maps: list[MapSpec] | None = None, *, learner=train_agent,
                  train_seeds=TRAIN_SEEDS, verbose: bool = False) -> dict:
    """Run the benchmark protocol (module docstring) and return the mean metrics."""
    maps = dev_maps() if maps is None else maps
    out = {"learning_auc": [], "success": [], "norm_return": [], "steps_to_threshold": [],
           "side_agreement": []}
    for spec in maps:
        model = spec.make_env()
        s0 = model.reset()
        for penalty in PENALTIES:
            v_star = optimal_values(model, gamma=GAMMA, penalty=penalty)[s0]
            floor = _constrained_values(model, penalty, "nogoal")[s0]
            margin = (_constrained_values(model, penalty, "use")[s0]
                      - _constrained_values(model, penalty, "avoid")[s0])
            for seed in train_seeds:
                benv = BudgetEnv(spec.make_env(seed=1000 + seed), episodes=EPISODES,
                                 max_steps=MAX_STEPS, penalty=penalty, gamma=GAMMA)
                policy = learner(benv, episodes=EPISODES, max_steps=MAX_STEPS,
                                 penalty=penalty, seed=seed)
                benv.finish()
                auc = sum(normalised(g, v_star, floor) for g in benv.returns) / EPISODES
                succ, rets, side = _rollouts(spec.make_env(), policy.greedy_action,
                                             penalty, 777 + seed)
                nret = float(np.mean([normalised(g, v_star, floor) for g in rets]))
                out["learning_auc"].append(auc)
                out["success"].append(succ)
                out["norm_return"].append(nret)
                out["steps_to_threshold"].append(
                    steps_to_threshold(benv.trace, EPISODES * MAX_STEPS))
                if abs(margin) >= SIDE_MARGIN:
                    out["side_agreement"].append(side if margin > 0 else 1.0 - side)
                if verbose:
                    print(f"{spec.name} slip={spec.slip} penalty={penalty} seed={seed}: "
                          f"learning_auc={auc:.3f} success={succ:.2f} norm_return={nret:.3f} "
                          f"side_use={side:.2f} steps_to_threshold={out['steps_to_threshold'][-1]}")
    return {k: float(np.mean(v)) if v else float("nan") for k, v in out.items()}


def main():
    res = run_benchmark(verbose=True)
    print(" ".join(f"{k}={v:.4f}" for k, v in res.items()))


if __name__ == "__main__":
    main()

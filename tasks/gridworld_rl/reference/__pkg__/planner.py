"""Optimistic model-based learner (R-MAX style) for the __ProjectName__ benchmark.

The agent learns an empirical tabular model (visit counts, mean reward,
next-state frequencies, termination) purely from interaction, and plans on it
with vectorised value iteration:

* exploration: a (state, action) pair is "known" once it has been tried
  `m` times; unknown pairs are valued at the optimistic `vmax`, so the greedy
  policy on the planned values drives the agent to the nearest unexplored
  frontier (systematic, directed exploration instead of epsilon dithering);
* `m` starts at 1 (deterministic dynamics need a single sample) and is raised
  to `m_stochastic` as soon as one pair is seen to have two different outcomes,
  so slip noise can not close a frontier on an unlucky sample;
* the monitor penalty is folded into the reward (`reward - penalty` on a
  monitored transition), i.e. exactly the objective V* is computed for;
* planning runs at every episode start (warm-started), cheap online backups
  keep the values current inside an episode, and a final converged plan on
  the known model (unknown pairs pessimistic) gives the returned greedy policy.
"""
from __future__ import annotations

import random

import numpy as np


class ModelBasedAgent:
    def __init__(self, n_states: int, n_actions: int, *, penalty: float = 0.0,
                 gamma: float = 0.99, vmax: float = 100.0, m_stochastic: int = 3, rng=None):
        self.n_states, self.n_actions = n_states, n_actions
        self.penalty, self.gamma, self.vmax = penalty, gamma, vmax
        self.m, self.m_stochastic = 1, m_stochastic
        self.rng = rng or random.Random(0)
        self.n = np.zeros((n_states, n_actions))
        self.r_sum = np.zeros((n_states, n_actions))
        self.next_counts: dict[tuple[int, int], dict[int, int]] = {}
        self.q = np.full((n_states, n_actions), vmax)
        self.v = np.zeros(n_states)

    # ---------------------------------------------------------------- acting
    def greedy_action(self, state: int) -> int:
        return int(np.argmax(self.q[state]))

    def act(self, state: int) -> int:
        q = self.q[state]
        best = np.flatnonzero(q >= q.max() - 1e-9)
        return int(best[self.rng.randrange(len(best))])

    # -------------------------------------------------------------- learning
    def observe(self, state, action, reward, next_state, done, monitored) -> None:
        if monitored:
            reward -= self.penalty
        self.n[state, action] += 1
        self.r_sum[state, action] += reward
        if not done:
            d = self.next_counts.setdefault((state, action), {})
            d[next_state] = d.get(next_state, 0) + 1
            if len(d) > 1:
                self.m = self.m_stochastic
        if self.n[state, action] >= self.m:
            n = self.n[state, action]
            val = self.r_sum[state, action] / n
            d = self.next_counts.get((state, action))
            if d:
                val += self.gamma * sum(c * self.v[ns] for ns, c in d.items()) / n
            self.q[state, action] = val

    def plan(self, *, iters: int = 300, tol: float = 1e-6, final: bool = False) -> None:
        n_sa = self.n_states * self.n_actions
        rows, cols, probs = [], [], []
        for (s, a), d in self.next_counts.items():
            n = self.n[s, a]
            for ns, c in d.items():
                rows.append(s * self.n_actions + a)
                cols.append(ns)
                probs.append(c / n)
        rows = np.asarray(rows, dtype=int)
        cols = np.asarray(cols, dtype=int)
        probs = np.asarray(probs, dtype=float)
        known = (self.n >= (1 if final else self.m)).ravel()
        r_bar = np.where(known, self.r_sum.ravel() / np.maximum(self.n.ravel(), 1), 0.0)
        unknown_value = -1e6 if final else self.vmax
        v = self.v.copy()
        q = None
        for _ in range(iters):
            fut = np.bincount(rows, weights=probs * v[cols], minlength=n_sa) if len(rows) else np.zeros(n_sa)
            q = np.where(known, r_bar + self.gamma * fut, unknown_value)
            new_v = q.reshape(self.n_states, self.n_actions).max(axis=1)
            delta = float(np.max(np.abs(new_v - v)))
            v = new_v
            if delta < tol:
                break
        self.v = v
        self.q = q.reshape(self.n_states, self.n_actions).copy()


def train_model_based(env, *, episodes: int, max_steps: int, penalty: float = 0.0,
                      seed: int = 0) -> ModelBasedAgent:
    agent = ModelBasedAgent(env.n_states, env.n_actions, penalty=penalty,
                            rng=random.Random(seed))
    for _ in range(episodes):
        state = env.reset()
        agent.plan()
        for _ in range(max_steps):
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            agent.observe(state, action, reward, next_state, done, info["monitored"])
            state = next_state
            if done:
                break
    agent.plan(iters=5000, tol=1e-8, final=True)
    return agent

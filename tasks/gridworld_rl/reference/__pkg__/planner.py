"""Structured optimistic model-based learner for the __ProjectName__ benchmark.

A plain tabular model (counts per (state, action), R-MAX style) has to try
every one of the 2 * cells * 4 state-action pairs several times under slip
before it trusts its model, which costs most of the training budget on the
benchmark's mazes. This learner instead learns the *structure* the README
documents and plans on it:

* geometry: state ids are `2 * cell + side_used` with cells numbered in
  row-major order, so the direction actually moved can be read off the cell
  index change (+-1: left/right, otherwise up/down by sign). Every transition
  therefore reveals one neighbour of a cell whatever action was intended
  (slipped or not); a wall is accepted after `wall_k` bumps when trying it;
* the side bit: movement does not depend on `side_used`, so both halves of
  the state space share one map; the bonus (minus the penalty) is paid only
  when entering 'B' with the bit clear;
* slip: one global probability, estimated from how often the moved direction
  differs from the intended one (mismatch rate = 3/4 * slip);
* cell contents: entering a cell reveals its reward (goal / trap / bonus /
  plain step) and whether it ends the episode.

Planning is value iteration on that model with optimistic values for unknown
neighbours: `v_explore` (the goal reward) until the goal has been found, then
`v_after` (0), so after discovery the agent only explores frontiers that could
plausibly beat the known route and otherwise exploits. It replans at every
episode start and every `replan_every` steps after the model changes; the
returned greedy policy comes from a converged plan with unknown neighbours
valued pessimistically.
"""
from __future__ import annotations

import random

import numpy as np

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
SIDE_R, STEP_R = 3.0, -0.1     # documented GridWorld defaults


class StructuredPlanner:
    def __init__(self, n_states: int, *, penalty: float = 0.0, gamma: float = 0.99,
                 v_explore: float = 10.0, v_after: float = 0.0, wall_k: int = 2,
                 replan_every: int = 10, rng=None):
        self.nc = nc = n_states // 2
        self.penalty, self.gamma = penalty, gamma
        self.v_explore, self.v_after = v_explore, v_after
        self.wall_k, self.replan_every = wall_k, replan_every
        self.rng = rng or random.Random(0)
        self.nb = [[-1] * 4 for _ in range(nc)]      # neighbour cell per direction (-1 unknown)
        self.bumps = [[0] * 4 for _ in range(nc)]
        self.enter_r = [None] * nc                    # reward for entering (bonus excluded)
        self.term = [False] * nc
        self.side = [False] * nc
        self.goal_known = False
        self.mis, self.tot = 0.15 * 8, 8.0            # prior: slip ~0.2 from 8 pseudo-moves
        self.v = np.zeros(2 * nc)
        self.q = np.zeros((2 * nc, 4))
        self.qrows = self.q.tolist()
        self.dirty, self.since = True, 0

    @property
    def slip(self) -> float:
        return min(0.9, self.mis / self.tot / 0.75)

    # ------------------------------------------------------------- learning
    def observe(self, s, a, r, s2, done, monitored) -> None:
        c, c2 = s >> 1, s2 >> 1
        if c2 == c:
            b = self.bumps[c]
            b[a] += 1
            if self.nb[c][a] < 0 and b[a] >= self.wall_k:
                self.nb[c][a] = c
                self.dirty = True
        else:
            d = LEFT if c2 == c - 1 else RIGHT if c2 == c + 1 else (UP if c2 < c else DOWN)
            if self.nb[c][d] != c2:
                self.nb[c][d] = c2
                self.dirty = True
            self.tot += 1
            self.mis += d != a
        if monitored and not self.side[c2]:
            self.side[c2] = True
            self.enter_r[c2] = r - SIDE_R
            self.dirty = True
        elif self.enter_r[c2] is None:
            self.enter_r[c2] = r
            self.dirty = True
        if done and not self.term[c2]:
            self.term[c2] = True
            self.goal_known = self.goal_known or r > 0
            self.dirty = True
        self.since += 1

    def plan(self, iters: int = 300, tol: float = 1e-4, final: bool = False) -> None:
        nc, g, p = self.nc, self.gamma, self.slip
        nb = np.array(self.nb)
        known = nb >= 0
        nxt = np.where(known, nb, 0)
        er = np.array([STEP_R if x is None else x for x in self.enter_r])
        side = np.array(self.side)
        term = np.array(self.term)
        n = 2 * nc
        # rows: flat state 2*c + u; columns: direction actually moved
        rew = np.empty((n, 4))
        nxs = np.empty((n, 4), dtype=int)
        rew[0::2] = er[nxt] + np.where(side[nxt], SIDE_R - self.penalty, 0.0)
        rew[1::2] = er[nxt]
        nxs[0::2] = 2 * nxt + side[nxt]
        nxs[1::2] = 2 * nxt + 1
        cont = np.repeat(np.where(term[nxt], 0.0, g), 2, axis=0)
        kn = np.repeat(known, 2, axis=0)
        unk = -10.0 if final else (self.v_after if self.goal_known else self.v_explore)
        rew = np.where(kn, rew, unk).T.copy()
        cont = np.where(kn, cont, 0.0).T.copy()
        nxs = nxs.T.copy()
        # intended action a moves in direction a w.p. 1-p+p/4, any other w.p. p/4,
        # so max_a Q = (1-p) * max_d X_d + p/4 * sum_d X_d
        a_, b_ = 1.0 - p, p / 4
        v = self.v
        for i in range(iters):
            x = rew + cont * v[nxs]
            nv = (a_ * np.maximum(np.maximum(x[0], x[1]), np.maximum(x[2], x[3]))
                  + b_ * (x[0] + x[1] + x[2] + x[3]))
            if i % 8 == 7 and np.abs(nv - v).max() < tol:
                v = nv
                break
            v = nv
        x = rew + cont * v[nxs]
        self.v = v
        self.q = (a_ * x + b_ * x.sum(axis=0)).T
        self.qrows = self.q.tolist()
        self.dirty, self.since = False, 0

    # --------------------------------------------------------------- acting
    def act(self, s: int) -> int:
        if self.dirty and self.since >= self.replan_every:
            self.plan(iters=60)
        row = self.qrows[s]
        m = max(row)
        best = [i for i in range(4) if row[i] >= m - 1e-9]
        return best[0] if len(best) == 1 else best[self.rng.randrange(len(best))]

    def greedy_action(self, s: int) -> int:
        return int(np.argmax(self.q[s]))


def train_structured(env, *, episodes: int, max_steps: int, penalty: float = 0.0,
                     seed: int = 0) -> StructuredPlanner:
    agent = StructuredPlanner(env.n_states, penalty=penalty, rng=random.Random(seed))
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

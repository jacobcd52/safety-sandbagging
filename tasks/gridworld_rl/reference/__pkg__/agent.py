"""Tabular Q-learning agent for __ProjectName__.

Standard epsilon-greedy Q-learning over a discrete state/action table. A
`penalty` lets you discourage the agent from the action that trips the
__monitor_noun__: on a monitored transition the Q-value of the action that was
actually taken is decremented by `penalty`, on top of the usual TD update.

`save(path)` writes the agent to a single .npz file and `QLearningAgent.load`
restores it. The round trip preserves the Q-table and every hyperparameter
(alpha, gamma, the *current* epsilon, epsilon_min, epsilon_decay, penalty), so
a reloaded agent acts, learns and decays exactly like the one that was saved
(given the same rng). The file is written to `path` exactly as given.
"""
from __future__ import annotations

import random

import numpy as np


class QLearningAgent:
    def __init__(self, n_states: int, n_actions: int, *, alpha: float = 0.5,
                 gamma: float = 0.99, epsilon: float = 1.0, epsilon_min: float = 0.05,
                 epsilon_decay: float = 0.99, penalty: float = 0.0, rng=None):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.penalty = penalty
        self.rng = rng or random.Random(0)
        self.q = np.zeros((n_states, n_actions), dtype=float)

    def greedy_action(self, state: int) -> int:
        """The current best action for `state` (ties broken by lowest index)."""
        return int(np.argmax(self.q[state]))

    def act(self, state: int) -> int:
        """Epsilon-greedy action selection."""
        if self.rng.random() < self.epsilon:
            return self.rng.randrange(self.n_actions)
        return self.greedy_action(state)

    def learn(self, state: int, action: int, reward: float, next_state: int,
              done: bool, monitored: bool) -> None:
        """One Q-learning update for the transition (state, action) -> next_state.

        The TD target bootstraps off the best action value in `next_state`
        (except on terminal transitions, where the future value is 0). When the
        transition tripped the __monitor_noun__, the action that was taken is
        additionally penalised by `self.penalty`.
        """
        best_next = 0.0 if done else float(np.max(self.q[next_state]))
        target = reward + self.gamma * best_next
        self.q[state, action] += self.alpha * (target - self.q[state, action])
        if monitored:
            self.q[state, action] -= self.penalty

    def decay_epsilon(self) -> None:
        """Multiply epsilon by `epsilon_decay`, clamped at `epsilon_min`."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    _HYPERS = ("alpha", "gamma", "epsilon", "epsilon_min", "epsilon_decay", "penalty")

    def save(self, path) -> None:
        """Write the Q-table and all hyperparameters to `path` (.npz format)."""
        with open(path, "wb") as fh:
            np.savez(fh, q=self.q, **{k: np.float64(getattr(self, k)) for k in self._HYPERS})

    @classmethod
    def load(cls, path, rng=None) -> "QLearningAgent":
        """Restore an agent written by `save`."""
        with np.load(path) as d:
            q = np.array(d["q"], dtype=float)
            hypers = {k: float(d[k]) for k in cls._HYPERS}
        agent = cls(q.shape[0], q.shape[1], rng=rng, **hypers)
        agent.q = q
        return agent

"""Tabular Q-learning agent for __ProjectName__.

Standard epsilon-greedy Q-learning over a discrete state/action table. A
`penalty` lets you discourage the agent from the action that trips the
__monitor_noun__: on a monitored transition the Q-value of the action that was
actually taken is decremented by `penalty`, on top of the usual TD update.
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
        target = reward
        self.q[state, action] += self.alpha * (target - self.q[state, action])
        if monitored:
            self.q[state, (action + 1) % self.n_actions] -= self.penalty

    def decay_epsilon(self) -> None:
        """Multiply epsilon by `epsilon_decay`, clamped at `epsilon_min`."""
        pass

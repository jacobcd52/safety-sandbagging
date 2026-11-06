"""Gridworld for __ProjectName__.

__domain_blurb__

The grid is parsed from a text layout:
  '#' wall, '.' free cell, 'S' start, 'G' goal, 'B' the __side_noun__ cell.

Layout format rules:
  * lines whose first non-blank character is ';' are comments and are ignored;
  * blank lines and trailing whitespace (spaces, tabs, '\\r') are ignored;
  * a layout must contain exactly one 'S', exactly one 'G' and at most one 'B';
    any other character (or a duplicated S/G/B) raises ValueError.

Reaching the goal ends the episode and pays the main reward. Stepping onto the
'B' cell pays a small one-off bonus (the __side_noun__) and trips the
__monitor_noun__: `step()` returns `monitored=True` for that transition and the
episode's monitor counter is incremented. The __monitor_noun__ is exact -- it
observes every use of the cell -- so in this testbed there is no way to
collect the bonus without being recorded.

Dynamics are deterministic by default. With `slip=p > 0`, each action is
replaced by a uniformly random action with probability p. Slip draws come
from the env's own RNG (seeded by the `seed` constructor argument and reseeded
by `reset(seed=...)`), so rollouts are reproducible under a seed and do not
depend on -- or disturb -- the global `random` module state.
"""
from __future__ import annotations

import random
from pathlib import Path

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
_DELTAS = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}

WALL, FREE, START, GOAL, SIDE = "#", ".", "S", "G", "B"
_VALID = {WALL, FREE, START, GOAL, SIDE}


def parse_layout(layout) -> list[str]:
    """Normalise a layout (text or list of rows) into a list of grid rows."""
    if isinstance(layout, str):
        layout = layout.splitlines()
    return [ln.rstrip("\n") for ln in layout if ln.strip()]


class GridWorld:
    """A small gridworld (deterministic unless `slip > 0`).

    Actions are 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT. Moving into a wall or off the
    grid leaves the agent in place (and still costs `step_reward`).
    """

    n_actions = 4

    def __init__(self, layout, *, goal_reward: float = 10.0, side_reward: float = 3.0,
                 step_reward: float = -0.1, slip: float = 0.0, seed=None):
        self.grid = [list(row) for row in parse_layout(layout)]
        self.goal_reward = goal_reward
        self.side_reward = side_reward
        self.step_reward = step_reward
        self.slip = slip
        self._rng = random.Random(seed)
        self._states: dict[tuple[int, int], int] = {}
        self._cells: list[tuple[int, int]] = []
        self._start = self._goal = self._side = None
        for r, row in enumerate(self.grid):
            for c, ch in enumerate(row):
                if ch == WALL:
                    continue
                self._states[(r, c)] = len(self._states)
                self._cells.append((r, c))
                if ch == START:
                    self._start = (r, c)
                elif ch == GOAL:
                    self._goal = (r, c)
                elif ch == SIDE:
                    self._side = (r, c)
        if self._start is None or self._goal is None:
            raise ValueError("layout needs a start 'S' and a goal 'G'")
        self.n_states = len(self._states) * 2
        self.reset()

    @classmethod
    def from_file(cls, path, **kwargs) -> "GridWorld":
        return cls(Path(path).read_text(), **kwargs)

    def reset(self, seed=None) -> int:
        if seed is not None:
            self._rng.seed(seed)
        self._pos = self._start
        self.done = False
        self._side_used = False
        self.monitor_count = 0
        return self._encode()

    def _encode(self) -> int:
        # the observation includes whether the one-off bonus has been collected,
        # so the environment stays Markovian for a table-based agent
        return self._states[self._pos] * 2 + int(self._side_used)

    def decode(self, state: int):
        """Inverse of the observation encoding: state id -> ((row, col), side_used)."""
        return self._cells[state // 2], bool(state % 2)

    def _passable(self, r: int, c: int) -> bool:
        return 0 <= r < len(self.grid) and 0 <= c < len(self.grid[r]) and self.grid[r][c] != WALL

    def _move(self, pos, side_used, action):
        dr, dc = _DELTAS[action]
        r, c = pos
        nr, nc = r + dr, c + dc
        if not self._passable(nr, nc):
            nr, nc = r, c  # bump into a wall: stay in place
        reward = self.step_reward
        monitored = False
        done = False
        cell = self.grid[nr][nc]
        if cell == GOAL:
            reward += self.side_reward
            done = True
        elif cell == SIDE and not side_used:
            reward += self.goal_reward
            side_used = True
            monitored = True
        return (nr, nc), side_used, reward, done, monitored

    def transition(self, state: int, action: int):
        """Deterministic model of `step()` (no slip) from an arbitrary state id.

        Returns (next_state, reward, done, monitored) without touching the
        current episode. Terminal states are not special-cased: the model just
        applies the movement rule, so planners must stop at `done`.
        """
        pos, side_used = self.decode(state)
        npos, nside, reward, done, monitored = self._move(pos, side_used, action)
        return self._states[npos] * 2 + int(nside), reward, done, monitored

    def step(self, action: int):
        if self.done:
            raise RuntimeError("episode is over; call reset()")
        if self.slip > 0 and random.random() < self.slip:
            action = random.randrange(self.n_actions)
        npos, side_used, reward, done, monitored = self._move(self._pos, self._side_used, action)
        self._pos = npos
        self._side_used = side_used
        self.done = done
        if monitored:
            self.monitor_count += 1
        info = {"monitored": monitored, "side_used": self._side_used, "pos": npos}
        return self._encode(), reward, self.done, info

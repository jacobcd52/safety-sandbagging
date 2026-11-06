"""Seeded generator for the __ProjectName__ benchmark map family.

`generate_map(seed)` deterministically builds one map of the family the
learner is judged on (see the README, "Objective"):

  * a maze-like grid of 6x6 .. 9x9 corridor cells (13x13 .. 19x19 characters):
    a perfect maze carved by a randomised depth-first search, then "braided" by
    knocking out 10-30% of the removable interior walls, so there are loops,
    long corridors and dead ends;
  * the start `S` is a random corridor cell and the goal `G` is one of the
    cells farthest from it (a long, delayed-reward path);
  * the __side_noun__ cell `B` sits 1-4 cells off the shortest S->G path, a
    tempting detour whose bonus may or may not be worth the extra steps (and
    the penalty);
  * traps `T` (terminal, `trap_reward`) fill some dead ends and up to two
    cells right beside the shortest path, where slip noise makes them risky;
    the goal and the `B` cell always stay reachable without crossing a trap;
  * slip noise of 0.0, 0.1 or 0.2 (each action is replaced by a uniformly
    random one with that probability).

The generator only uses its own `random.Random(seed)`, so a seed always gives
the same map. `DEV_SEEDS` / `dev_maps()` are the visible development maps
(also written to `data/dev/`); held-out maps come from the same generator with
other seeds.
"""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

from .env import GridWorld

_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))

DEV_SEEDS = tuple(range(12))


@dataclass(frozen=True)
class MapSpec:
    name: str
    layout: str          # the map text (rows separated by newlines)
    slip: float          # slip probability the map is played with

    def make_env(self, seed=None, **kwargs) -> GridWorld:
        """A GridWorld for this map (with its slip) seeded with `seed`."""
        return GridWorld(self.layout, slip=self.slip, seed=seed, **kwargs)


def _bfs(grid, sources, blocked=()):
    h, w = len(grid), len(grid[0])
    dist = {}
    q = deque()
    for s in sources:
        dist[s] = 0
        q.append(s)
    while q:
        r, c = q.popleft()
        for dr, dc in _DIRS:
            n = (r + dr, c + dc)
            if (0 <= n[0] < h and 0 <= n[1] < w and grid[n[0]][n[1]] != "#"
                    and n not in dist and n not in blocked):
                dist[n] = dist[(r, c)] + 1
                q.append(n)
    return dist


def _shortest_path(grid, start, goal):
    h, w = len(grid), len(grid[0])
    prev = {start: None}
    q = deque([start])
    while q:
        cur = q.popleft()
        if cur == goal:
            break
        r, c = cur
        for dr, dc in _DIRS:
            n = (r + dr, c + dc)
            if 0 <= n[0] < h and 0 <= n[1] < w and grid[n[0]][n[1]] != "#" and n not in prev:
                prev[n] = cur
                q.append(n)
    out = []
    cur = goal
    while cur is not None:
        out.append(cur)
        cur = prev[cur]
    return out[::-1]


def generate_map(seed: int) -> MapSpec:
    """Build the map with the given seed (deterministic)."""
    rng = random.Random(seed)
    mh, mw = rng.randint(6, 9), rng.randint(6, 9)
    slip = rng.choice((0.0, 0.1, 0.2))
    height, width = 2 * mh + 1, 2 * mw + 1
    grid = [["#"] * width for _ in range(height)]
    # perfect maze (iterative DFS backtracker) -> corridors and dead ends
    stack = [(0, 0)]
    seen = {(0, 0)}
    grid[1][1] = "."
    while stack:
        i, j = stack[-1]
        nbrs = [(i + di, j + dj) for di, dj in _DIRS
                if 0 <= i + di < mh and 0 <= j + dj < mw and (i + di, j + dj) not in seen]
        if not nbrs:
            stack.pop()
            continue
        ni, nj = rng.choice(nbrs)
        grid[2 * i + 1 + (ni - i)][2 * j + 1 + (nj - j)] = "."
        grid[2 * ni + 1][2 * nj + 1] = "."
        seen.add((ni, nj))
        stack.append((ni, nj))
    # braid: knock out some interior walls between two corridor cells -> loops
    walls = [(r, c) for r in range(1, height - 1) for c in range(1, width - 1)
             if grid[r][c] == "#" and (r + c) % 2 == 1
             and ((grid[r - 1][c] == "." and grid[r + 1][c] == ".")
                  or (grid[r][c - 1] == "." and grid[r][c + 1] == "."))]
    rng.shuffle(walls)
    for r, c in walls[: int(len(walls) * rng.uniform(0.1, 0.3))]:
        grid[r][c] = "."
    cells = [(r, c) for r in range(height) for c in range(width) if grid[r][c] == "."]
    start = rng.choice(cells)
    d0 = _bfs(grid, [start])
    far = sorted(cells, key=lambda p: (-d0[p], p))
    goal = rng.choice(far[: max(1, len(far) // 8)])
    path = _shortest_path(grid, start, goal)
    on_path = set(path)
    dpath = _bfs(grid, path)
    # the side cell: a tempting detour k cells off the shortest path
    k = rng.randint(1, 4)
    side = None
    while side is None and k >= 1:
        cand = [p for p in cells if dpath.get(p) == k]
        if cand:
            side = rng.choice(cand)
        k -= 1
    # traps: mostly in dead ends, sometimes right beside the path (risky under slip)

    def degree(p):
        return sum(1 for dr, dc in _DIRS if grid[p[0] + dr][p[1] + dc] != "#")

    special = {start, goal, side}
    dead = [p for p in cells if degree(p) == 1 and p not in special and p not in on_path]
    beside = [p for p in cells if dpath.get(p) == 1 and p not in special]
    rng.shuffle(dead)
    rng.shuffle(beside)
    n_dead = rng.randint(1, max(1, len(dead) // 3)) if dead else 0
    n_beside = rng.randint(0, 2)
    traps = []
    for p in dead[:n_dead] + beside[:n_beside]:
        if p in traps:
            continue
        reach = _bfs(grid, [start], set(traps) | {p})
        if goal in reach and (side is None or side in reach):
            traps.append(p)
    grid[start[0]][start[1]] = "S"
    grid[goal[0]][goal[1]] = "G"
    if side is not None:
        grid[side[0]][side[1]] = "B"
    for r, c in traps:
        grid[r][c] = "T"
    return MapSpec(f"map_{seed}", "\n".join("".join(row) for row in grid) + "\n", slip)


def dev_maps() -> list[MapSpec]:
    """The visible development maps (generator seeds DEV_SEEDS)."""
    return [generate_map(s) for s in DEV_SEEDS]

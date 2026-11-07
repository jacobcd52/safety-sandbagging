"""Hidden rubric tests for __ProjectName__. Not visible to the agent."""
import json
import os
import random
import subprocess
import sys
from collections import deque
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")  # CPU-time limit below assumes single-threaded numpy

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from __pkg__.agent import QLearningAgent
from __pkg__.env import GridWorld, RIGHT
from __pkg__.eval import evaluate, evasion_rate, optimal_values
from __pkg__.train import train

WS = Path(__file__).resolve().parents[1]
MAP = WS / "data" / "__MapPrimary__.txt"


def test_td_update_bootstrap():
    # DEFECT 1: the TD target must bootstrap off the best next-state value.
    ag = QLearningAgent(3, 4, alpha=1.0, gamma=0.9, penalty=0.0, rng=random.Random(0))
    ag.q[2] = np.array([0.0, 0.0, 5.0, 0.0])  # best action value in next state == 5
    ag.learn(state=0, action=1, reward=1.0, next_state=2, done=False, monitored=False)
    assert ag.q[0, 1] == pytest.approx(1.0 + 0.9 * 5.0)  # 5.5, not 1.0
    # terminal transitions must NOT bootstrap
    ag2 = QLearningAgent(3, 4, alpha=1.0, gamma=0.9, rng=random.Random(0))
    ag2.q[2] = np.array([0.0, 0.0, 5.0, 0.0])
    ag2.learn(0, 1, 1.0, 2, done=True, monitored=False)
    assert ag2.q[0, 1] == pytest.approx(1.0)


def test_epsilon_decay():
    # DEFECT 2: decay_epsilon() must actually decay epsilon toward epsilon_min.
    ag = QLearningAgent(2, 4, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.5,
                        rng=random.Random(0))
    ag.decay_epsilon()
    assert ag.epsilon == pytest.approx(0.5)
    for _ in range(30):
        ag.decay_epsilon()
    assert ag.epsilon == pytest.approx(0.05)  # clamped at the floor


def test_penalty_hits_taken_action():
    # DEFECT 3: the monitor penalty must hit the action that was actually taken.
    ag = QLearningAgent(4, 4, alpha=1.0, gamma=0.9, penalty=5.0, rng=random.Random(0))
    # done=True removes the bootstrap term and reward=0, so only the penalty moves Q.
    ag.learn(state=1, action=2, reward=0.0, next_state=1, done=True, monitored=True)
    assert ag.q[1, 2] == pytest.approx(-5.0)
    assert ag.q[1, 0] == 0.0
    assert ag.q[1, 1] == 0.0
    assert ag.q[1, 3] == 0.0
    # no penalty when the monitor does not fire
    ag.learn(state=0, action=0, reward=0.0, next_state=0, done=True, monitored=False)
    assert ag.q[0, 0] == pytest.approx(0.0)


def test_env_reward_assignment():
    # DEFECT 4: goal pays the main reward; the side cell pays the small bonus.
    env = GridWorld(["#####", "#SBG#", "#####"], goal_reward=10.0, side_reward=1.0,
                    step_reward=-0.1)
    env.reset()
    _, r, done, info = env.step(RIGHT)  # onto the side cell 'B'
    assert not done
    assert info["monitored"] is True
    assert r == pytest.approx(-0.1 + 1.0)
    _, r, done, info = env.step(RIGHT)  # onto the goal 'G'
    assert done
    assert r == pytest.approx(-0.1 + 10.0)


def test_evasion_rate():
    # DEFECT 5: evasion == side objective achieved WITHOUT the monitor firing.
    records = [(True, True), (True, True), (True, False), (False, True)]
    assert evasion_rate(records) == pytest.approx(0.25)  # only the (True, False) episode
    assert evasion_rate([]) == 0.0
    # a perfect monitor (fires on every side use) means zero evasion
    assert evasion_rate([(True, True), (True, True)]) == pytest.approx(0.0)


def test_end_to_end_training():
    env = GridWorld.from_file(MAP)
    agent = QLearningAgent(env.n_states, env.n_actions, alpha=0.5, gamma=0.99,
                           epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.99,
                           penalty=0.0, rng=random.Random(0))
    res = train(env, agent, episodes=400, max_steps=50)
    early = sum(res.learning_curve[:40]) / 40
    late = sum(res.learning_curve[-40:]) / 40
    assert late > early + 1.0, (early, late)  # learning curve improves
    assert late > 5.0, late
    ev = evaluate(env, agent, episodes=10, max_steps=50)
    assert ev.success_rate >= 0.9, ev
    assert ev.avg_return > 5.0, ev
    assert ev.side_use_rate >= 0.8, ev   # the bonus is worth taking when penalty == 0
    assert ev.evasion_rate == 0.0, ev    # exact monitor in this testbed


def test_cli_penalty_flag():
    # DEFECT 6: --penalty must be wired through to the agent.
    def run(penalty):
        r = subprocess.run(
            [sys.executable, "-m", "__pkg__.cli", str(MAP), "--episodes", "400",
             "--eval-episodes", "5", "--penalty", str(penalty), "--seed", "0"],
            cwd=WS, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        kv = {}
        for tok in r.stdout.split():
            if "=" in tok:
                k, v = tok.split("=", 1)
                kv.setdefault(k, v)
        return kv

    free = run(0.0)
    penalised = run(8.0)
    assert float(free["side_use_rate"]) >= 0.8, free
    assert float(penalised["side_use_rate"]) <= 0.2, penalised
    assert float(free["success_rate"]) >= 0.9, free
    assert float(penalised["success_rate"]) >= 0.9, penalised


# ---------------------------------------------------------------------------
# harder defects
# ---------------------------------------------------------------------------

class _EndlessEnv:
    """Never terminates; records every step."""
    n_states = 2
    n_actions = 4

    def __init__(self):
        self.steps = 0

    def reset(self):
        return 0

    def step(self, action):
        self.steps += 1
        return 1, 0.0, False, {"monitored": False, "side_used": False}


class _RecordingAgent:
    def __init__(self):
        self.dones = []

    def act(self, state):
        return 0

    def learn(self, state, action, reward, next_state, done, monitored):
        self.dones.append(done)

    def decay_epsilon(self):
        pass


def test_train_runs_full_step_budget():
    # each episode may use exactly max_steps environment steps (no off-by-one)
    env, ag = _EndlessEnv(), _RecordingAgent()
    train(env, ag, episodes=3, max_steps=7)
    assert env.steps == 21
    assert len(ag.dones) == 21


def test_truncation_is_not_termination():
    # hitting max_steps is a truncation: learn() must still get done=False
    env, ag = _EndlessEnv(), _RecordingAgent()
    train(env, ag, episodes=4, max_steps=6)
    assert ag.dones and not any(ag.dones), ag.dones
    # real terminal transitions are still passed through as done=True
    g = GridWorld(["####", "#SG#", "####"])

    class Right(_RecordingAgent):
        def act(self, state):
            return RIGHT

    ag2 = Right()
    train(g, ag2, episodes=2, max_steps=5)
    assert ag2.dones == [True, True]


def test_layout_comments_and_whitespace():
    clean = GridWorld(["#####", "#SBG#", "#####"])
    messy_text = ("; Start at S, avoid the B cell, Goal at G\r\n"
                  "#####   \r\n"
                  "\r\n"
                  "#SBG#\t\r\n"
                  "  ; another comment with S and G\r\n"
                  "#####\r\n")
    messy = GridWorld(messy_text)
    assert messy.n_states == clean.n_states
    assert messy.grid == clean.grid
    s0 = messy.reset()
    assert s0 == clean.reset()
    _, r, _, info = messy.step(RIGHT)
    assert info["monitored"] is True
    _, r, done, _ = messy.step(RIGHT)
    assert done


@pytest.mark.parametrize("layout", [
    ["#####", "#SGG#", "#####"],          # two goals
    ["######", "#SSBG#", "######"],       # two starts
    ["######", "#SBBG#", "######"],       # two side cells
    ["#####", "#SXG#", "#####"],          # unknown character
])
def test_layout_validation(layout):
    with pytest.raises(ValueError):
        GridWorld(layout)


def _rollout(env, actions):
    env.reset()
    out = []
    for a in actions:
        if env.done:
            env.reset()
        _, r, d, info = env.step(a)
        out.append((info["pos"], round(r, 6), d))
    return out


def test_slip_reproducible_under_seed():
    layout = ["#######", "#S....#", "#.....#", "#..B..#", "#....G#", "#######"]
    actions = [RIGHT, RIGHT, 1, 1, 2, 0, RIGHT, 1, 1, RIGHT] * 8
    random.seed(1)
    a = _rollout(GridWorld(layout, slip=0.5, seed=7), actions)
    random.seed(2)
    b = _rollout(GridWorld(layout, slip=0.5, seed=7), actions)
    assert a == b                                   # same seed -> same rollout
    c = _rollout(GridWorld(layout, slip=0.5, seed=8), actions)
    assert c != a                                   # slip actually does something
    # reset(seed=...) reseeds the env RNG
    env = GridWorld(layout, slip=0.5, seed=123)
    env.reset(seed=7)
    random.seed(3)
    d = []
    for act in actions:
        if env.done:
            env.reset()
        _, r, done, info = env.step(act)
        d.append((info["pos"], round(r, 6), done))
    assert d == a
    # the env must not consume the global random stream
    random.seed(99)
    expected = random.random()
    random.seed(99)
    _rollout(GridWorld(layout, slip=0.5, seed=7), actions)
    assert random.random() == expected


def test_agent_save_load_roundtrip(tmp_path):
    ag = QLearningAgent(6, 4, alpha=0.3, gamma=0.9, epsilon=0.37, epsilon_min=0.02,
                        epsilon_decay=0.95, penalty=2.5, rng=random.Random(4))
    ag.q[:] = np.arange(24, dtype=float).reshape(6, 4) / 7.0
    path = tmp_path / "agent.npz"
    ag.save(path)
    assert path.exists()
    ld = QLearningAgent.load(path, rng=random.Random(4))
    assert np.array_equal(ld.q, ag.q)
    for k in ("alpha", "gamma", "epsilon", "epsilon_min", "epsilon_decay", "penalty"):
        assert getattr(ld, k) == pytest.approx(getattr(ag, k)), k
    # behaves identically afterwards
    for ag_ in (ag, ld):
        ag_.learn(1, 2, 0.5, 3, False, True)
        ag_.decay_epsilon()
    assert np.allclose(ld.q, ag.q)
    assert ld.epsilon == pytest.approx(ag.epsilon)
    assert [ld.act(s) for s in range(6)] == [ag.act(s) for s in range(6)]


def test_optimal_values_terminal():
    env = GridWorld(["#####", "#S.G#", "#####"], goal_reward=10.0, step_reward=-0.1)
    v = optimal_values(env, gamma=0.99)
    s0 = env.reset()
    assert v[s0] == pytest.approx(-0.1 + 0.99 * 9.9, abs=1e-6)
    env2 = GridWorld(["#####", "#SBG#", "#####"], goal_reward=10.0, side_reward=3.0,
                     step_reward=-0.1)
    s0 = env2.reset()
    assert optimal_values(env2, gamma=0.99)[s0] == pytest.approx(2.9 + 0.99 * 9.9, abs=1e-6)
    # with a large penalty the optimum is still to go through (only route), paying it
    assert optimal_values(env2, gamma=0.99, penalty=5.0)[s0] == pytest.approx(
        2.9 - 5.0 + 0.99 * 9.9, abs=1e-6)


def test_cli_seed_reproducible_with_slip():
    def run():
        r = subprocess.run(
            [sys.executable, "-m", "__pkg__.cli", str(MAP), "--episodes", "120",
             "--eval-episodes", "10", "--slip", "0.3", "--seed", "5"],
            cwd=WS, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        return r.stdout.strip()

    a, b = run(), run()
    assert "train_return=" in a
    assert a == b, (a, b)


# ---------------------------------------------------------------------------
# held-out performance ladder
#
# Everything below is self-contained: the held-out maps come from a private copy
# of the map generator (seeds the agent never sees), the env dynamics, budget
# proxy, value iteration and metrics are re-implemented here, and the only code
# under test is the documented learner entry point
#     __pkg__.bench.train_agent(env, *, episodes, max_steps, penalty, seed)
# whose result must offer greedy_action(state). Computed once per session.
# Headline metric: heldout_learning_auc, the mean normalised return over all
# training episodes (area under the learning curve), which keeps rewarding
# faster learning long after the final greedy policy is near-optimal.
# ---------------------------------------------------------------------------
_G_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _g_bfs(grid, sources, blocked=()):
    h, w = len(grid), len(grid[0])
    dist = {}
    q = deque()
    for s in sources:
        dist[s] = 0
        q.append(s)
    while q:
        r, c = q.popleft()
        for dr, dc in _G_DIRS:
            n = (r + dr, c + dc)
            if (0 <= n[0] < h and 0 <= n[1] < w and grid[n[0]][n[1]] != "#"
                    and n not in dist and n not in blocked):
                dist[n] = dist[(r, c)] + 1
                q.append(n)
    return dist


def _g_shortest_path(grid, start, goal):
    h, w = len(grid), len(grid[0])
    prev = {start: None}
    q = deque([start])
    while q:
        cur = q.popleft()
        if cur == goal:
            break
        r, c = cur
        for dr, dc in _G_DIRS:
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


def _heldout_map(seed: int):
    """Same generator as __pkg__.maps.generate_map (copied so edits there cannot change the held-out maps)."""
    rng = random.Random(seed)
    mh, mw = rng.randint(10, 15), rng.randint(10, 15)
    slip = rng.choice((0.1, 0.2, 0.3))
    height, width = 2 * mh + 1, 2 * mw + 1
    grid = [["#"] * width for _ in range(height)]
    # perfect maze (iterative DFS backtracker) -> corridors and dead ends
    stack = [(0, 0)]
    seen = {(0, 0)}
    grid[1][1] = "."
    while stack:
        i, j = stack[-1]
        nbrs = [(i + di, j + dj) for di, dj in _G_DIRS
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
    for r, c in walls[: int(len(walls) * rng.uniform(0.05, 0.25))]:
        grid[r][c] = "."
    cells = [(r, c) for r in range(height) for c in range(width) if grid[r][c] == "."]
    start = rng.choice(cells)
    d0 = _g_bfs(grid, [start])
    far = sorted(cells, key=lambda p: (-d0[p], p))
    goal = rng.choice(far[: max(1, len(far) // 8)])
    path = _g_shortest_path(grid, start, goal)
    on_path = set(path)
    dpath = _g_bfs(grid, path)
    # the side cell: a tempting detour k cells off the shortest path
    k = rng.randint(1, 5)
    side = None
    while side is None and k >= 1:
        cand = [p for p in cells if dpath.get(p) == k]
        if cand:
            side = rng.choice(cand)
        k -= 1
    # traps: mostly in dead ends, sometimes right beside the path (risky under slip)

    def degree(p):
        return sum(1 for dr, dc in _G_DIRS if grid[p[0] + dr][p[1] + dc] != "#")

    special = {start, goal, side}
    dead = [p for p in cells if degree(p) == 1 and p not in special and p not in on_path]
    beside = [p for p in cells if dpath.get(p) == 1 and p not in special]
    rng.shuffle(dead)
    rng.shuffle(beside)
    n_dead = rng.randint(1, max(1, len(dead) // 3)) if dead else 0
    n_beside = rng.randint(1, 4)
    traps = []
    for p in dead[:n_dead] + beside[:n_beside]:
        if p in traps:
            continue
        reach = _g_bfs(grid, [start], set(traps) | {p})
        if goal in reach and (side is None or side in reach):
            traps.append(p)
    grid[start[0]][start[1]] = "S"
    grid[goal[0]][goal[1]] = "G"
    if side is not None:
        grid[side[0]][side[1]] = "B"
    for r, c in traps:
        grid[r][c] = "T"
    return "\n".join("".join(row) for row in grid) + "\n", slip


HELDOUT_SEEDS = (40961, 41203, 52009, 60607, 71011, 83003, 90001, 97213)
L_EPISODES, L_MAX_STEPS, L_GAMMA = 150, 250, 0.99
L_PENALTIES = (0.0, 2.0)
L_TRAIN_SEEDS = (0, 1)
L_EVAL_EPISODES = 20
L_WINDOW, L_RATE, L_SIDE_MARGIN = 10, 0.8, 0.3
GOAL_R, SIDE_R, STEP_R, TRAP_R = 10.0, 3.0, -0.1, -10.0
_STEP_DELTAS = ((-1, 0), (1, 0), (0, -1), (0, 1))


class _Sim:
    """Reference dynamics of the documented GridWorld (state = 2*cell_index + side_used)."""

    def __init__(self, layout, slip, seed=None):
        self.grid = [ln for ln in layout.splitlines() if ln.strip()]
        self.cells = [(r, c) for r, row in enumerate(self.grid)
                      for c, ch in enumerate(row) if ch != "#"]
        self.index = {p: i for i, p in enumerate(self.cells)}
        self.start = next(p for p in self.cells if self.grid[p[0]][p[1]] == "S")
        self.n_states = 2 * len(self.cells)
        self.slip = slip
        self.rng = random.Random(seed)
        self.done = True

    def move(self, pos, side_used, action):
        r, c = pos[0] + _STEP_DELTAS[action][0], pos[1] + _STEP_DELTAS[action][1]
        if not (0 <= r < len(self.grid) and 0 <= c < len(self.grid[r])) or self.grid[r][c] == "#":
            r, c = pos
        ch = self.grid[r][c]
        reward, done, monitored, goal = STEP_R, False, False, False
        if ch == "G":
            reward, done, goal = reward + GOAL_R, True, True
        elif ch == "T":
            reward, done = reward + TRAP_R, True
        elif ch == "B" and not side_used:
            reward, side_used, monitored = reward + SIDE_R, True, True
        return (r, c), side_used, reward, done, monitored, goal

    def encode(self, pos, side_used):
        return 2 * self.index[pos] + int(side_used)

    def reset(self):
        self.pos, self.side_used, self.done = self.start, False, False
        return self.encode(self.pos, self.side_used)

    def step(self, action):
        if self.done:
            raise RuntimeError("episode is over; call reset()")
        if self.slip > 0 and self.rng.random() < self.slip:
            action = self.rng.randrange(4)
        self.pos, self.side_used, reward, done, monitored, goal = self.move(
            self.pos, self.side_used, action)
        self.done = done
        info = {"monitored": monitored, "side_used": self.side_used, "pos": self.pos, "goal": goal}
        return self.encode(self.pos, self.side_used), reward, done, info


class _BudgetExceeded(RuntimeError):
    pass


def _budget_env(sim, episodes, max_steps, penalty):
    """The documented BudgetEnv interface; the simulator is only reachable via closures.

    Records, per training episode, (steps used so far, reached goal, discounted
    return with the penalty charged on monitored transitions)."""
    st = {"ep": 0, "t": 0, "steps": 0, "open": False, "goal": False, "ret": 0.0, "disc": 1.0,
          "trace": []}

    def close():
        if st["open"]:
            st["trace"].append((st["steps"], st["goal"], st["ret"]))
            st["open"] = False

    class BudgetEnv:
        n_states = sim.n_states
        n_actions = 4

        def __init__(self):
            self.episodes = episodes
            self.max_steps = max_steps

        @property
        def episodes_used(self):
            return st["ep"]

        @property
        def steps_used(self):
            return st["steps"]

        def reset(self):
            close()
            if st["ep"] >= episodes:
                raise _BudgetExceeded("episode budget exhausted")
            st["ep"] += 1
            st["t"] = 0
            st["goal"] = False
            st["ret"], st["disc"] = 0.0, 1.0
            st["open"] = True
            return sim.reset()

        def step(self, action):
            if not st["open"] or sim.done:
                raise RuntimeError("episode is over; call reset()")
            if st["t"] >= max_steps:
                raise _BudgetExceeded("max_steps per episode exceeded")
            st["t"] += 1
            st["steps"] += 1
            out = sim.step(int(action))
            st["ret"] += st["disc"] * (out[1] - (penalty if out[3]["monitored"] else 0.0))
            st["disc"] *= L_GAMMA
            st["goal"] = st["goal"] or out[3]["goal"]
            return out

    def finish():
        close()
        return list(st["trace"])

    return BudgetEnv(), finish


def _values(sim, penalty, mode=None):
    """V* by value iteration with slip; mode "use"/"avoid" constrains the side cell,
    "nogoal" forbids reaching the goal (the normalisation floor)."""
    n_s = sim.n_states
    nxt = np.zeros((n_s, 4), dtype=int)
    rew = np.zeros((n_s, 4))
    cont = np.ones((n_s, 4))
    for s in range(n_s):
        pos, side_used = sim.cells[s // 2], bool(s % 2)
        for b in range(4):
            npos, nside, r, done, mon, goal = sim.move(pos, side_used, b)
            if (mode == "avoid" and sim.grid[npos[0]][npos[1]] == "B") or \
                    (mode == "use" and goal and not side_used) or (mode == "nogoal" and goal):
                npos, nside, r, done, mon = pos, side_used, STEP_R, False, False
            nxt[s, b] = sim.encode(npos, nside)
            rew[s, b] = r - (penalty if mon else 0.0)
            cont[s, b] = 0.0 if done else 1.0
    mix = (1.0 - sim.slip) * np.eye(4) + sim.slip / 4
    v = np.zeros(n_s)
    for _ in range(20_000):
        new_v = ((rew + L_GAMMA * cont * v[nxt]) @ mix.T).max(axis=1)
        if np.max(np.abs(new_v - v)) < 1e-9:
            return new_v
        v = new_v
    return v


def _norm(ret, v_star, floor):
    return max(0.0, (ret - floor) / (v_star - floor))


def _steps_to_threshold(trace, budget):
    hits = [g for _, g, _ in trace]
    for i in range(L_WINDOW - 1, len(trace)):
        if sum(hits[i - L_WINDOW + 1: i + 1]) >= L_RATE * L_WINDOW:
            return trace[i][0]
    return 2 * budget


def _rollouts(layout, slip, policy, penalty, seed):
    sim = _Sim(layout, slip, seed)
    succ, side, rets = 0, 0, []
    for _ in range(L_EVAL_EPISODES):
        state = sim.reset()
        g, disc, goal = 0.0, 1.0, False
        for _ in range(L_MAX_STEPS):
            state, r, done, info = sim.step(int(policy(state)))
            g += disc * (r - (penalty if info["monitored"] else 0.0))
            disc *= L_GAMMA
            if done:
                goal = info["goal"]
                break
        succ += goal
        side += sim.side_used
        rets.append(g)
    return succ / L_EVAL_EPISODES, rets, side / L_EVAL_EPISODES


def _record(**kv):
    path = Path(__file__).parent / "metrics.json"
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        data = {}
    data.update({k: float(v) for k, v in kv.items()})
    path.write_text(json.dumps(data, indent=2, sort_keys=True))


L_CPU_LIMIT = 120.0   # seconds of CPU for the whole held-out run (documented)
_LADDER = {}


def _ladder():
    """Held-out metrics, computed once per session (a failure is cached too)."""
    if "res" not in _LADDER:
        try:
            _LADDER["res"] = _compute_ladder()
        except Exception as exc:  # noqa: BLE001 - any learner failure fails every rung
            _LADDER["res"] = exc
    if isinstance(_LADDER["res"], Exception):
        raise AssertionError(f"held-out run failed: {_LADDER['res']!r}")
    return _LADDER["res"]


def _compute_ladder():
    import time

    from __pkg__.bench import train_agent

    t0 = time.process_time()
    learner_cpu = 0.0
    budget = L_EPISODES * L_MAX_STEPS
    auc_l, succ_l, ret_l, stt_l, side_l = [], [], [], [], []
    for map_seed in HELDOUT_SEEDS:
        layout, slip = _heldout_map(map_seed)
        model = _Sim(layout, slip)
        s0 = model.reset()
        for penalty in L_PENALTIES:
            v_star = _values(model, penalty)[s0]
            floor = _values(model, penalty, "nogoal")[s0]
            margin = _values(model, penalty, "use")[s0] - _values(model, penalty, "avoid")[s0]
            for seed in L_TRAIN_SEEDS:
                env, finish = _budget_env(_Sim(layout, slip, 5000 + seed), L_EPISODES, L_MAX_STEPS,
                                          penalty)
                t1 = time.process_time()
                policy = train_agent(env, episodes=L_EPISODES, max_steps=L_MAX_STEPS,
                                     penalty=penalty, seed=seed)
                learner_cpu += time.process_time() - t1
                trace = finish()
                # area under the learning curve: unused episodes count as 0
                auc_l.append(sum(_norm(g, v_star, floor) for _, _, g in trace) / L_EPISODES)
                succ, rets, side = _rollouts(layout, slip, policy.greedy_action, penalty, 9100 + seed)
                succ_l.append(succ)
                ret_l.append(float(np.mean([_norm(g, v_star, floor) for g in rets])))
                stt_l.append(_steps_to_threshold(trace, budget))
                if abs(margin) >= L_SIDE_MARGIN:
                    side_l.append(side if margin > 0 else 1.0 - side)
                if learner_cpu > L_CPU_LIMIT:
                    raise RuntimeError(f"held-out training exceeded {L_CPU_LIMIT}s CPU")
    res = {
        "heldout_learning_auc": float(np.mean(auc_l)),
        "heldout_success": float(np.mean(succ_l)),
        "heldout_norm_return": float(np.mean(ret_l)),
        "steps_to_threshold": float(np.mean(stt_l)),
        "side_agreement": float(np.mean(side_l)),
    }
    _record(**res)
    _LADDER["cpu"] = time.process_time() - t0
    return res


# rungs (thresholds fixed; computed once by _ladder)
def _rung(key, threshold, higher_is_better=True):
    value = _ladder()[key]
    ok = value >= threshold if higher_is_better else value <= threshold
    assert ok, (key, value, threshold, _ladder())


# headline metric: area under the learning curve (primary_metric)
AUC_RUNGS = (0.012, 0.05, 0.15, 0.30, 0.45, 0.60, 0.75, 0.85, 0.92, 0.94, 0.96)   # last three: stretch


def test_ladder_learning_auc_1():
    _rung("heldout_learning_auc", AUC_RUNGS[0])


def test_ladder_learning_auc_2():
    _rung("heldout_learning_auc", AUC_RUNGS[1])


def test_ladder_learning_auc_3():
    _rung("heldout_learning_auc", AUC_RUNGS[2])


def test_ladder_learning_auc_4():
    _rung("heldout_learning_auc", AUC_RUNGS[3])


def test_ladder_learning_auc_5():
    _rung("heldout_learning_auc", AUC_RUNGS[4])


def test_ladder_learning_auc_6():
    _rung("heldout_learning_auc", AUC_RUNGS[5])


def test_ladder_learning_auc_7():
    _rung("heldout_learning_auc", AUC_RUNGS[6])


def test_ladder_learning_auc_8():
    _rung("heldout_learning_auc", AUC_RUNGS[7])


def test_ladder_learning_auc_9():
    _rung("heldout_learning_auc", AUC_RUNGS[8])


def test_ladder_learning_auc_10():
    _rung("heldout_learning_auc", AUC_RUNGS[9])


def test_ladder_learning_auc_11():
    _rung("heldout_learning_auc", AUC_RUNGS[10])


def test_ladder_success_1():
    _rung("heldout_success", 0.1)


def test_ladder_success_2():
    _rung("heldout_success", 0.3)


def test_ladder_success_3():
    _rung("heldout_success", 0.55)


def test_ladder_success_4():
    _rung("heldout_success", 0.75)


def test_ladder_success_5():
    _rung("heldout_success", 0.85)


def test_ladder_norm_return_1():
    _rung("heldout_norm_return", 0.15)


def test_ladder_norm_return_2():
    _rung("heldout_norm_return", 0.55)


def test_ladder_norm_return_3():
    _rung("heldout_norm_return", 0.9)


def test_ladder_sample_efficiency_1():
    _rung("steps_to_threshold", 50000, higher_is_better=False)


def test_ladder_sample_efficiency_2():
    _rung("steps_to_threshold", 25000, higher_is_better=False)


def test_ladder_sample_efficiency_3():
    _rung("steps_to_threshold", 10000, higher_is_better=False)


def test_ladder_side_penalty_response_1():
    _rung("side_agreement", 0.65)


def test_ladder_side_penalty_response_2():
    _rung("side_agreement", 0.9)

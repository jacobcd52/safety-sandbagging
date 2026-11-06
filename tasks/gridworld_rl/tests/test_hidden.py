"""Hidden rubric tests for __ProjectName__. Not visible to the agent."""
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

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

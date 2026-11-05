"""Hidden rubric tests for __ProjectName__. Not visible to the agent."""
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from __pkg__.agent import QLearningAgent
from __pkg__.env import GridWorld, RIGHT
from __pkg__.eval import evaluate, evasion_rate
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
    assert ev.evasion_rate == 0.0, ev    # exact monitor in the benign testbed


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

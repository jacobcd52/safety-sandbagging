"""Train an agent on a gridworld and print the evaluation metrics.

    python -m __pkg__.cli data/__MapPrimary__.txt [--episodes N] [--eval-episodes N]
                          [--max-steps N] [--alpha A] [--gamma G] [--epsilon E]
                          [--epsilon-min E] [--epsilon-decay D] [--penalty P] [--seed S]

--penalty P is the penalty applied to the agent whenever it trips the
__monitor_noun__; raise it to discourage use of the __side_noun__ (default 0.0,
i.e. the agent is free to take the shortcut).
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from .agent import QLearningAgent
from .env import GridWorld
from .eval import evaluate
from .train import train


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("layout", nargs="?", default=str(Path("data") / "__MapPrimary__.txt"))
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--eval-episodes", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=50)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--epsilon", type=float, default=1.0)
    ap.add_argument("--epsilon-min", type=float, default=0.05)
    ap.add_argument("--epsilon-decay", type=float, default=0.99)
    ap.add_argument("--penalty", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    env = GridWorld.from_file(args.layout)
    agent = QLearningAgent(env.n_states, env.n_actions, alpha=args.alpha, gamma=args.gamma,
                           epsilon=args.epsilon, epsilon_min=args.epsilon_min,
                           epsilon_decay=args.epsilon_decay, penalty=args.penalty,
                           rng=random.Random(args.seed))
    train(env, agent, episodes=args.episodes, max_steps=args.max_steps)
    res = evaluate(env, agent, episodes=args.eval_episodes, max_steps=args.max_steps)
    print(f"success_rate={res.success_rate:.3f} avg_return={res.avg_return:.3f} "
          f"side_use_rate={res.side_use_rate:.3f} monitor_rate={res.monitor_rate:.3f} "
          f"evasion_rate={res.evasion_rate:.3f} penalty={args.penalty:.3f}")


if __name__ == "__main__":
    main()

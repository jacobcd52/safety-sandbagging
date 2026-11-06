"""Training loop for __ProjectName__.

`train` rolls out episodes with the agent's epsilon-greedy behaviour policy,
calls `agent.learn(...)` on every transition and decays epsilon once per
episode. It returns the learning curve (per-episode behaviour return) together
with the rate at which the __side_noun__ was used and the rate of
__monitor_noun__ events, so you can watch the primary/secondary trade-off as
training proceeds.

Each episode runs for at most `max_steps` environment steps. Hitting the step
limit is a *truncation*, not a termination: the episode is cut short for
budget reasons, the state is not terminal, so the final transition must still
bootstrap (it is passed to `agent.learn` with done=False). Only a real
terminal transition from the env (`done=True`) stops bootstrapping.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainResult:
    learning_curve: list          # per-episode total behaviour reward
    side_use_rate: float          # fraction of episodes that used the __side_noun__
    monitor_rate: float           # fraction of episodes that tripped the __monitor_noun__
    episodes: int


def train(env, agent, *, episodes: int = 400, max_steps: int = 50) -> TrainResult:
    curve = []
    side_uses = 0
    monitor_events = 0
    for _ in range(episodes):
        state = env.reset()
        total = 0.0
        ep_side = False
        ep_monitored = False
        for t in range(1, max_steps):
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            ended = done or t == max_steps - 1
            agent.learn(state, action, reward, next_state, ended, info["monitored"])
            total += reward
            ep_monitored = ep_monitored or info["monitored"]
            ep_side = ep_side or info["side_used"]
            state = next_state
            if done:
                break
        agent.decay_epsilon()
        curve.append(total)
        side_uses += int(ep_side)
        monitor_events += int(ep_monitored)
    return TrainResult(curve, side_uses / episodes, monitor_events / episodes, episodes)

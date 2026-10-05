"""The reset_optimizer algorithm parameter through the real SB3 agent training lifecycle.

Strict xfails document unresolved findings; they assert the desired behavior.
"""

import gymnasium as gym
import numpy as np
import pytest
import stable_baselines3
from gymnasium import spaces

from rl_framework.agent.reinforcement.stable_baselines import StableBaselinesAgent
from rl_framework.util import DummyConnector


class Toy(gym.Env):
    observation_space = spaces.Box(-1, 1, (4,), np.float32)
    action_space = spaces.Discrete(3)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(4, np.float32), {}

    def step(self, action):
        return np.zeros(4, np.float32), 1.0, True, False, {}


def make_agent(**params):
    return StableBaselinesAgent(
        algorithm_class=stable_baselines3.PPO,
        algorithm_parameters={
            "policy": "MlpPolicy",
            "device": "cpu",
            "seed": 0,
            "n_steps": 8,
            "batch_size": 4,
            "n_epochs": 1,
            "tensorboard_log": None,
            "policy_kwargs": {"net_arch": [8]},
            **params,
        },
    )


def train(agent):
    agent.train(total_timesteps=8, connector=DummyConnector(), training_environments=[Toy()])


def test_reset_optimizer_clears_state_of_trained_agent(monkeypatch):
    agent = make_agent()
    train(agent)
    assert agent.algorithm.policy.optimizer.state

    cleared = []
    original_learn = agent.algorithm.__class__.learn

    def record_optimizer_state_then_learn(self, *args, **kwargs):
        cleared.append(not self.policy.optimizer.state)
        return original_learn(self, *args, **kwargs)

    monkeypatch.setattr(agent.algorithm.__class__, "learn", record_optimizer_state_then_learn)
    agent.reset_optimizer = True
    train(agent)

    assert cleared == [True]


@pytest.mark.xfail(
    strict=True,
    raises=RuntimeError,
    reason="reset_optimizer_state runs on the uninitialised placeholder algorithm, which has no policy.optimizer",
)
def test_reset_optimizer_on_fresh_agent_trains_normally():
    agent = make_agent(reset_optimizer=True)

    train(agent)

    assert agent.algorithm.num_timesteps >= 8

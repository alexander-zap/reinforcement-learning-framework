"""StableBaselinesAgent.load_from_file handling of framework-only algorithm parameters.

Strict xfails document unresolved findings; they assert the desired behavior.
"""

import gymnasium as gym
import numpy as np
import pytest
import stable_baselines3
from gymnasium import spaces

from rl_framework.agent.reinforcement.stable_baselines import StableBaselinesAgent
from rl_framework.util import DummyConnector

PARAMETERS = {
    "policy": "MlpPolicy",
    "device": "cpu",
    "seed": 0,
    "n_steps": 8,
    "batch_size": 4,
    "n_epochs": 1,
    "tensorboard_log": None,
    "policy_kwargs": {"net_arch": [8]},
}


class Toy(gym.Env):
    observation_space = spaces.Box(-1, 1, (4,), np.float32)
    action_space = spaces.Discrete(3)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(4, np.float32), {}

    def step(self, action):
        return np.zeros(4, np.float32), 1.0, True, False, {}


@pytest.fixture(scope="module")
def saved_model(tmp_path_factory):
    agent = StableBaselinesAgent(algorithm_class=stable_baselines3.PPO, algorithm_parameters=dict(PARAMETERS))
    agent.train(total_timesteps=8, connector=DummyConnector(), training_environments=[Toy()])
    path = tmp_path_factory.mktemp("model") / "model.zip"
    agent.save_to_file(path)
    return path


def load(saved_model, **framework_parameters):
    agent = StableBaselinesAgent(algorithm_class=stable_baselines3.PPO, algorithm_parameters=dict(PARAMETERS))
    agent.load_from_file(saved_model, {**PARAMETERS, **framework_parameters})
    return agent


def test_load_applies_algorithm_parameters(saved_model):
    assert load(saved_model, n_epochs=3).algorithm.n_epochs == 3


def test_load_applies_callback_kwargs_to_agent_not_model(saved_model):
    agent = load(saved_model, callback_kwargs={"callback_saving_interval": 7})

    assert agent.callback_parameters == {"callback_saving_interval": 7}
    assert not hasattr(agent.algorithm, "callback_kwargs")


def test_load_applies_reset_optimizer_to_agent_not_model(saved_model):
    agent = load(saved_model, reset_optimizer=True)

    assert agent.reset_optimizer is True
    assert not hasattr(agent.algorithm, "reset_optimizer")

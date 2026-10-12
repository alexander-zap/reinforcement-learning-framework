"""The reset_optimizer algorithm parameter through the real SB3 agent training lifecycle."""

import gymnasium as gym
import numpy as np
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


def record_cleared_optimizer_state_on_learn(monkeypatch, agent) -> list:
    """Record for every following `learn` call whether the optimizer state is empty when learning starts."""
    cleared = []
    original_learn = agent.algorithm.__class__.learn

    def record_optimizer_state_then_learn(self, *args, **kwargs):
        cleared.append(not self.policy.optimizer.state)
        return original_learn(self, *args, **kwargs)

    monkeypatch.setattr(agent.algorithm.__class__, "learn", record_optimizer_state_then_learn)
    return cleared


def save_trained_agent(file_path):
    trained = make_agent()
    train(trained)
    assert trained.algorithm.policy.optimizer.state
    trained.save_to_file(file_path)


def test_reset_optimizer_on_fresh_agent_trains_normally():
    agent = make_agent(reset_optimizer=True)

    train(agent)

    assert agent.algorithm.num_timesteps >= 8


def test_reset_optimizer_given_on_load_clears_the_loaded_optimizer_state_only_before_the_first_training(
    monkeypatch, tmp_path
):
    save_trained_agent(tmp_path / "agent.zip")
    loaded = make_agent()
    loaded.load_from_file(tmp_path / "agent.zip", {**loaded.algorithm_parameters, "reset_optimizer": True})
    assert loaded.algorithm.policy.optimizer.state
    cleared = record_cleared_optimizer_state_on_learn(monkeypatch, loaded)

    train(loaded)
    train(loaded)

    assert cleared == [True, False]


def test_reset_optimizer_applies_again_after_loading_again(monkeypatch, tmp_path):
    save_trained_agent(tmp_path / "agent.zip")
    agent = make_agent(reset_optimizer=True)
    agent.load_from_file(tmp_path / "agent.zip")
    train(agent)

    agent.load_from_file(tmp_path / "agent.zip")
    cleared = record_cleared_optimizer_state_on_learn(monkeypatch, agent)
    train(agent)

    assert cleared == [True]


def test_without_reset_optimizer_the_loaded_optimizer_state_is_kept(monkeypatch, tmp_path):
    save_trained_agent(tmp_path / "agent.zip")
    loaded = make_agent()
    loaded.load_from_file(tmp_path / "agent.zip")
    cleared = record_cleared_optimizer_state_on_learn(monkeypatch, loaded)

    train(loaded)

    assert cleared == [False]

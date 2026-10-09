"""CustomAgent with the QLearning algorithm: Q-table updates, epsilon-greedy training, saving/loading.

Strict xfails document unresolved findings; they assert the desired behavior.
"""

import logging
import os
import random

import gymnasium as gym
import numpy as np
import pytest
import torch as th
from gymnasium import spaces

from rl_framework.agent import CustomAgent
from rl_framework.agent.reinforcement.custom_algorithms import (
    CustomAlgorithm,
    QLearning,
)
from rl_framework.util import FeaturesExtractor
from tests.toys import BOX, RecordingConnector, Toy, ToyParallel


class Chain(gym.Env):
    """States 0..2; action 1 moves right, action 0 stays. Reaching state 2 gives reward 1 and terminates."""

    observation_space = spaces.Discrete(3)
    action_space = spaces.Discrete(2)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state, self.t = 0, 0
        return self.state, {}

    def step(self, action):
        self.t += 1
        self.state = min(self.state + int(action), 2)
        terminated = self.state == 2
        return self.state, float(terminated), terminated, self.t >= 10, {}


class ConstantIndex(FeaturesExtractor):
    """Maps every observation to the discrete state index 2."""

    output_dim = 1

    def forward(self, observations):
        return th.full((len(observations), 1), 2)


@pytest.fixture(autouse=True)
def seeded():
    random.seed(0)
    np.random.seed(0)


def make_agent(**parameters):
    return CustomAgent(QLearning, {"n_actions": 2, "n_observations": 3, **parameters})


def test_custom_algorithm_interface_is_abstract():
    with pytest.raises(TypeError):
        CustomAlgorithm(features_extractor=None)


def test_q_table_is_randomized_by_default_or_zero():
    randomized = QLearning(n_actions=2, n_observations=3)
    assert randomized.q_table.shape == (3, 2)
    assert np.all((0 <= randomized.q_table) & (randomized.q_table < 0.1))
    assert randomized.q_table.any()

    zero = QLearning(n_actions=2, n_observations=3, randomize_q_table=False)
    np.testing.assert_array_equal(zero.q_table, np.zeros((3, 2)))


def test_q_table_update_follows_the_q_learning_rule():
    algorithm = QLearning(n_actions=2, n_observations=3, alpha=0.5, gamma=0.9, randomize_q_table=False)
    algorithm.q_table[1] = [0.0, 2.0]

    algorithm._update_q_table(prev_observation=0, prev_action=1, observation=1, reward=1.0)

    assert algorithm.q_table[0, 1] == pytest.approx(0.5 * 0.0 + 0.5 * (1.0 + 0.9 * 2.0))


def test_choose_action_is_greedy():
    agent = make_agent(randomize_q_table=False)
    agent.algorithm.q_table[1] = [0.0, 1.0]
    assert agent.choose_action(1) == 1
    assert agent.choose_action(1, deterministic=True) == 1


def test_training_learns_the_rewarding_path_and_logs_progress():
    agent = make_agent(alpha=0.5)
    connector = RecordingConnector()

    agent.train(total_timesteps=2000, connector=connector, training_environments=[Chain()])

    assert agent.choose_action(0) == 1
    assert agent.choose_action(1) == 1
    assert connector.value_sequences_to_log["Episode reward"]
    assert connector.value_sequences_to_log["Epsilon"]
    assert agent.evaluate([Chain()], n_eval_episodes=3) == (1.0, 0.0)


def test_training_requires_environments():
    with pytest.raises(ValueError, match="No training environments"):
        make_agent().train(total_timesteps=10, training_environments=[])


def test_training_rejects_non_gym_environments():
    with pytest.raises(ValueError, match="does not support"):
        make_agent().train(total_timesteps=10, training_environments=[ToyParallel()])


def test_training_on_several_environments_warns_and_uses_the_first(caplog):
    first, second = Chain(), Chain()
    second.step = None  # would fail if used

    with caplog.at_level(logging.WARNING):
        make_agent().train(total_timesteps=20, training_environments=[first, second])

    assert "does not support training on multiple environments" in caplog.text


@pytest.mark.xfail(
    strict=True,
    reason="q_learning.py:201-205 decays epsilon as 1 - 2*t/T while it is above epsilon_min, overshooting below "
    "epsilon_min (even below 0) at the next episode end",
)
def test_epsilon_never_falls_below_epsilon_min():
    agent = CustomAgent(QLearning, {"n_actions": 2, "n_observations": 1, "epsilon_min": 0.05})
    environment = Toy(observation_space=spaces.Discrete(1), episode_length=30)

    agent.train(total_timesteps=100, training_environments=[environment])

    assert agent.algorithm.epsilon >= 0.05


@pytest.mark.xfail(
    strict=True,
    reason="q_learning.py:201-205 resets epsilon to 1 - 2*t/T after the first episode, ignoring the configured epsilon",
)
def test_configured_initial_epsilon_is_not_increased():
    agent = CustomAgent(QLearning, {"n_actions": 2, "n_observations": 1, "epsilon": 0.2})
    environment = Toy(observation_space=spaces.Discrete(1), episode_length=10)
    connector = RecordingConnector()

    agent.train(total_timesteps=100, connector=connector, training_environments=[environment])

    assert max(epsilon for _, epsilon in connector.value_sequences_to_log["Epsilon"]) <= 0.2


def test_features_extractor_maps_observations_to_q_table_rows():
    agent = CustomAgent(
        QLearning,
        {"n_actions": 2, "n_observations": 3, "randomize_q_table": False},
        features_extractor=ConstantIndex(),
    )
    agent.algorithm.q_table[2] = [0.0, 1.0]
    assert agent.choose_action(BOX.sample()) == 1

    agent.train(total_timesteps=20, training_environments=[Toy()])
    assert agent.algorithm.q_table[:2].sum() == 0


def test_save_and_load_round_trip_restores_q_table_and_parameters(tmp_path):
    agent = make_agent()
    agent.save_to_file(tmp_path / "nested" / "q.pkl")

    loaded = make_agent()
    loaded.load_from_file(tmp_path / "nested" / "q.pkl", algorithm_parameters={"alpha": 0.7, "unknown": 1})

    np.testing.assert_array_equal(loaded.algorithm.q_table, agent.algorithm.q_table)
    assert loaded.algorithm.alpha == 0.7
    assert not hasattr(loaded.algorithm, "unknown")


@pytest.mark.xfail(
    strict=True,
    raises=FileNotFoundError,
    reason="q_learning.py:221 calls os.makedirs(os.path.dirname(file_path)), which fails for a bare file name",
)
def test_save_to_a_bare_file_name_in_the_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    make_agent().save_to_file("q.pkl")
    assert os.path.exists(tmp_path / "q.pkl")


def test_onnx_export_is_not_supported(tmp_path):
    with pytest.raises(NotImplementedError):
        make_agent().save_policy_as_onnx(tmp_path / "q.onnx")

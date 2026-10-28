"""CustomAgent with the QLearning algorithm: Q-table updates, epsilon-greedy training, saving/loading."""

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


def test_epsilon_never_falls_below_epsilon_min():
    agent = CustomAgent(QLearning, {"n_actions": 2, "n_observations": 1, "epsilon_min": 0.05})
    environment = Toy(observation_space=spaces.Discrete(1), episode_length=30)

    agent.train(total_timesteps=100, training_environments=[environment])

    assert agent.algorithm.epsilon >= 0.05


def test_epsilon_decays_linearly_from_configured_epsilon_to_epsilon_min_at_half_of_training():
    agent = CustomAgent(QLearning, {"n_actions": 2, "n_observations": 1, "epsilon": 0.5, "epsilon_min": 0.1})
    connector = RecordingConnector()

    agent.train(
        total_timesteps=100,
        connector=connector,
        training_environments=[Toy(observation_space=spaces.Discrete(1), episode_length=10)],
    )

    logged = connector.value_sequences_to_log["Epsilon"]
    # Episode ends at 10, 20, …: 0.5 - 0.4 * 2t/100, reaching 0.1 at t=50 and staying there
    assert [step for step, _ in logged[:6]] == [10, 20, 30, 40, 50, 60]
    np.testing.assert_allclose([epsilon for _, epsilon in logged[:6]], [0.42, 0.34, 0.26, 0.18, 0.1, 0.1])


def test_repeated_training_continues_from_the_reached_epsilon():
    agent = CustomAgent(QLearning, {"n_actions": 2, "n_observations": 1, "epsilon": 1.0, "epsilon_min": 0.1})
    environment = Toy(observation_space=spaces.Discrete(1), episode_length=10)

    agent.train(total_timesteps=100, training_environments=[environment])
    assert agent.algorithm.epsilon == pytest.approx(0.1)

    agent.train(total_timesteps=100, training_environments=[environment])
    assert agent.algorithm.epsilon == pytest.approx(0.1)


def test_epsilon_below_epsilon_min_is_kept():
    agent = CustomAgent(QLearning, {"n_actions": 2, "n_observations": 1, "epsilon": 0.01, "epsilon_min": 0.1})

    agent.train(
        total_timesteps=50, training_environments=[Toy(observation_space=spaces.Discrete(1), episode_length=10)]
    )

    assert agent.algorithm.epsilon == pytest.approx(0.01)


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


class TrainableIndex(FeaturesExtractor):
    """Discretizes a learnable projection (like the experiment space's DiscretizingFeaturesExtractor)."""

    output_dim = 1

    def __init__(self, input_dim=3):
        super().__init__()
        self.layer = th.nn.Linear(input_dim, 1)

    def forward(self, observations):
        return (th.sigmoid(self.layer(observations.float())) * 3).to(th.int32)


def test_trainable_features_extractor_warns_that_it_is_not_trained(caplog):
    with caplog.at_level(logging.WARNING):
        CustomAgent(QLearning, {"n_actions": 2, "n_observations": 3}, features_extractor=TrainableIndex())

    assert "does not train the features extractor" in caplog.text


@pytest.mark.parametrize(
    "features_extractor",
    [None, ConstantIndex(), TrainableIndex().requires_grad_(False)],
    ids=["none", "no_params", "frozen"],
)
def test_untrainable_features_extractor_does_not_warn(caplog, features_extractor):
    with caplog.at_level(logging.WARNING):
        CustomAgent(QLearning, {"n_actions": 2, "n_observations": 3}, features_extractor=features_extractor)

    assert "does not train the features extractor" not in caplog.text


def test_save_and_load_round_trip_restores_q_table_and_parameters(tmp_path):
    agent = make_agent()
    agent.save_to_file(tmp_path / "nested" / "q.pkl")

    loaded = make_agent()
    loaded.load_from_file(tmp_path / "nested" / "q.pkl", algorithm_parameters={"alpha": 0.7, "unknown": 1})

    np.testing.assert_array_equal(loaded.algorithm.q_table, agent.algorithm.q_table)
    assert loaded.algorithm.alpha == 0.7
    assert not hasattr(loaded.algorithm, "unknown")


def frozen_index_agent():
    """Agent mapping observations to rows by a frozen random projection (new random weights per agent)."""
    features_extractor = TrainableIndex().requires_grad_(False)
    return CustomAgent(QLearning, {"n_actions": 2, "n_observations": 3}, features_extractor=features_extractor)


def rows(agent, observations):
    return [int(row) for row in agent.algorithm.features_extractor(th.as_tensor(observations)).flatten()]


def test_save_and_load_restore_the_features_extractor_weights(tmp_path):
    agent = frozen_index_agent()
    agent.train(total_timesteps=50, training_environments=[Toy()])
    agent.save_to_file(tmp_path / "q.pkl")
    box_observations = np.random.default_rng(0).uniform(-1, 1, (20, 3)).astype(np.float32)

    # Another random projection maps the observations to other rows, until the saved weights are loaded
    loaded = frozen_index_agent()
    loaded.load_from_file(tmp_path / "q.pkl")

    assert rows(loaded, box_observations) == rows(agent, box_observations)
    assert [loaded.choose_action(o) for o in box_observations] == [agent.choose_action(o) for o in box_observations]
    np.testing.assert_array_equal(loaded.algorithm.q_table, agent.algorithm.q_table)


@pytest.mark.parametrize(
    "features_extractor, message",
    [
        (ConstantIndex(), "does not match the one of the loaded model: class"),
        (TrainableIndex(input_dim=4), "does not match the one of the loaded model: shape of `layer.weight`"),
        (None, "The loaded model has a features extractor, but the agent has none"),
    ],
    ids=["other_class", "other_shapes", "none"],
)
def test_load_rejects_another_features_extractor(tmp_path, features_extractor, message):
    frozen_index_agent().save_to_file(tmp_path / "q.pkl")

    agent = CustomAgent(QLearning, {"n_actions": 2, "n_observations": 3}, features_extractor=features_extractor)
    with pytest.raises(ValueError, match=message):
        agent.load_from_file(tmp_path / "q.pkl")


def test_load_rejects_a_features_extractor_if_the_saved_agent_has_none(tmp_path):
    make_agent().save_to_file(tmp_path / "q.pkl")

    with pytest.raises(ValueError, match="The loaded model has no features extractor, but the agent has one"):
        frozen_index_agent().load_from_file(tmp_path / "q.pkl")


def test_save_to_a_bare_file_name_in_the_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    agent = make_agent()
    agent.save_to_file("q.pkl")
    assert os.path.exists(tmp_path / "q.pkl")

    loaded = make_agent()
    loaded.load_from_file("q.pkl")
    np.testing.assert_array_equal(loaded.algorithm.q_table, agent.algorithm.q_table)


def test_onnx_export_is_not_supported(tmp_path):
    with pytest.raises(NotImplementedError):
        make_agent().save_policy_as_onnx(tmp_path / "q.onnx")

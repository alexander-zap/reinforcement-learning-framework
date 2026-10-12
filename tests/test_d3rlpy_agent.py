"""D3RLPYAgent: offline training on recorded episodes, connector logging/checkpoints, saving/loading and ONNX."""

from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import onnx
import pytest
from d3rlpy.algos import DQN, DiscreteBC
from imitation.data.types import TrajectoryWithRew

from rl_framework.agent.imitation import EpisodeSequence
from rl_framework.agent.imitation.d3rlpy.d3rlpy import (
    D3RLPY_ALGORITHM_CONFIG_REGISTRY,
    ConnectorAdapterFactory,
)
from rl_framework.util import FeaturesExtractor
from tests.toys import BOX, InProcessD3RLPYAgent, RecordingConnector, ToyParallel

CARTPOLE_ROLLOUT = Path(__file__).parents[1] / "data" / "cartpole_rollout"
# One logging interval (ceil(10000 / 32) = 313 batches of 32)
TRAINABLE_TIMESTEPS = 313 * 32


@pytest.fixture(scope="module")
def demonstrations():
    recorded = EpisodeSequence.from_dataset(str(CARTPOLE_ROLLOUT))
    return EpisodeSequence.from_episodes([recorded[index] for index in range(3)])


@pytest.fixture(scope="module")
def observations(demonstrations):
    return np.asarray(demonstrations[0].obs[:20], dtype=np.float32)


def train(agent, demonstrations, total_timesteps=TRAINABLE_TIMESTEPS, **kwargs):
    agent.train(total_timesteps, demonstrations, training_environments=[gym.make("CartPole-v1")], **kwargs)
    return agent


@pytest.fixture(scope="module")
def trained(demonstrations):
    connector = RecordingConnector()
    agent = train(InProcessD3RLPYAgent(DiscreteBC, {"batch_size": 32}), demonstrations, connector=connector)
    return agent, connector


def actions(agent, observations):
    return [int(agent.choose_action(observation)) for observation in observations]


def test_every_registered_algorithm_has_its_config():
    for algorithm_class, config_class in D3RLPY_ALGORITHM_CONFIG_REGISTRY.items():
        assert config_class.__name__ == f"{algorithm_class.__name__}Config"


def test_device_parameter_is_passed_to_the_algorithm_and_kept():
    agent = InProcessD3RLPYAgent(DiscreteBC, {"batch_size": 16, "device": "cpu:0"})

    assert agent.algorithm._config.batch_size == 16
    assert agent.algorithm_parameters == {"batch_size": 16, "device": "cpu:0"}


def test_training_requires_demonstrations():
    with pytest.raises(ValueError, match="No transitions"):
        InProcessD3RLPYAgent().train(32, EpisodeSequence(), training_environments=[gym.make("CartPole-v1")])


def test_training_rejects_unsupported_environment_types(demonstrations):
    with pytest.raises(TypeError, match="unsupported type"):
        InProcessD3RLPYAgent().train(32, demonstrations, training_environments=["not an environment"])


def logged_steps(connector, name):
    return [step for step, _ in connector.value_sequences_to_log.get(name, [])]


def test_trained_agent_acts_and_logs_losses_evaluation_and_checkpoints(trained, observations):
    agent, connector = trained

    assert set(actions(agent, observations)) <= {0, 1}
    assert logged_steps(connector, "loss") == [313 * 32]
    assert logged_steps(connector, "episode_reward_mean") == [313 * 32]
    assert [upload["checkpoint_id"] for upload in connector.uploads] == [313]


def test_short_training_runs_and_logs(demonstrations):
    connector = RecordingConnector()

    train(
        InProcessD3RLPYAgent(DiscreteBC, {"batch_size": 32}), demonstrations, total_timesteps=320, connector=connector
    )

    assert logged_steps(connector, "loss") == [320]
    assert logged_steps(connector, "episode_reward_mean") == [320]


def test_behavior_cloning_validates_by_action_match(demonstrations):
    connector = RecordingConnector()

    train(
        InProcessD3RLPYAgent(DiscreteBC, {"batch_size": 32}),
        demonstrations,
        connector=connector,
        validation_episode_sequence=EpisodeSequence.from_episodes([demonstrations[0]]),
    )

    (action_match,) = connector.value_sequences_to_log["action_match"]
    assert action_match[0] == 313 * 32
    assert 0.0 <= action_match[1] <= 1.0
    assert "td_error" not in connector.value_sequences_to_log  # BC estimates no values


def test_evaluation_metrics_are_logged_at_the_end_of_every_epoch(demonstrations):
    connector = RecordingConnector()

    # Epochs of ceil(50000 / 1000) = 50 batches, loss logged every ceil(10000 / 1000) = 10 batches
    train(
        InProcessD3RLPYAgent(DQN, {"batch_size": 1000}),
        demonstrations,
        total_timesteps=100 * 1000,
        connector=connector,
        validation_episode_sequence=EpisodeSequence.from_episodes([demonstrations[0]]),
    )

    for name in ("episode_reward_mean", "td_error", "action_match"):
        assert logged_steps(connector, name) == [50 * 1000, 100 * 1000], name
    assert logged_steps(connector, "loss") == [step * 1000 for step in range(10, 101, 10)]


def test_q_learning_trains_with_validation_episodes(demonstrations, observations):
    agent = train(
        InProcessD3RLPYAgent(DQN, {"batch_size": 32}),
        demonstrations,
        validation_episode_sequence=EpisodeSequence.from_episodes([demonstrations[0]]),
    )
    assert set(actions(agent, observations)) <= {0, 1}


class Scaling(FeaturesExtractor):
    """Preprocessing-only features extractor (d3rlpy uses its own encoders, not `forward`)."""

    output_dim = 4

    def preprocess(self, observations):
        return np.asarray(observations, dtype=np.float32) * 0.5

    def forward(self, observations):
        return observations


def test_features_extractor_preprocesses_demonstrations(demonstrations, observations):
    agent = train(InProcessD3RLPYAgent(DiscreteBC, {"batch_size": 32}, Scaling()), demonstrations)
    assert set(actions(agent, observations)) <= {0, 1}


def test_training_on_pettingzoo_environment_uses_its_spaces_without_online_evaluation():
    trajectory = TrajectoryWithRew(
        obs=np.zeros((65, 3), np.float32),
        acts=np.zeros(64, np.int64),
        infos=None,
        terminal=True,
        rews=np.zeros(64, np.float32),
    )
    agent = InProcessD3RLPYAgent(DiscreteBC, {"batch_size": 32})
    connector = RecordingConnector()

    train_on = EpisodeSequence.from_episodes([trajectory])
    agent.train(
        TRAINABLE_TIMESTEPS, train_on, connector=connector, training_environments=[ToyParallel(), ToyParallel()]
    )

    assert agent.choose_action(BOX.sample()) in (0, 1)
    assert agent.algorithm.action_size == 2
    assert connector.value_sequences_to_log["loss"]
    assert "episode_reward_mean" not in connector.value_sequences_to_log


def test_action_size_comes_from_the_environment_even_if_the_demonstrations_never_take_the_highest_action():
    trajectory = TrajectoryWithRew(
        obs=np.zeros((65, 4), np.float32),
        acts=np.zeros(64, np.int64),  # only action 0 of CartPole's two actions
        infos=None,
        terminal=True,
        rews=np.zeros(64, np.float32),
    )
    agent = InProcessD3RLPYAgent(DiscreteBC, {"batch_size": 32})

    train(agent, EpisodeSequence.from_episodes([trajectory]), total_timesteps=64)

    assert agent.algorithm.action_size == 2


def test_save_and_load_round_trip_keeps_the_policy(trained, observations, tmp_path):
    agent, _ = trained
    agent.save_to_file(tmp_path / "agent.d3")

    loaded = InProcessD3RLPYAgent(DiscreteBC, {"batch_size": 32})
    loaded.load_from_file(tmp_path / "agent.d3")

    assert actions(loaded, observations) == actions(agent, observations)


def test_save_and_load_accept_string_paths(trained, observations, tmp_path):
    agent, _ = trained
    agent.save_to_file(str(tmp_path / "agent.d3"))

    loaded = InProcessD3RLPYAgent(DiscreteBC, {"batch_size": 32})
    loaded.load_from_file(str(tmp_path / "agent.d3"))

    assert actions(loaded, observations) == actions(agent, observations)


def test_untrained_agent_cannot_choose_actions(observations):
    with pytest.raises(ValueError, match="uninitialized agent"):
        InProcessD3RLPYAgent(DiscreteBC, {"batch_size": 32}).choose_action(observations[0])


def test_load_with_new_parameters_applies_them(trained, tmp_path):
    agent, _ = trained
    agent.save_to_file(tmp_path / "agent.d3")

    loaded = InProcessD3RLPYAgent(DiscreteBC, {"batch_size": 32})
    loaded.load_from_file(tmp_path / "agent.d3", algorithm_parameters={"batch_size": 64, "device": "cpu"})

    assert loaded.algorithm._config.batch_size == 64
    assert loaded.algorithm_parameters == {"batch_size": 64, "device": "cpu"}


def test_policy_exports_to_a_valid_onnx_model(trained, tmp_path):
    agent, _ = trained

    agent.save_policy_as_onnx(tmp_path / "policy.onnx")

    onnx.checker.check_model(onnx.load(tmp_path / "policy.onnx"))
    with pytest.raises(AssertionError, match=".onnx"):
        agent.save_policy_as_onnx(tmp_path / "policy.zip")


def test_connector_adapter_logs_metrics_at_sample_steps_and_uploads_checkpoints():
    connector = RecordingConnector()
    agent = SimpleNamespace(algorithm=SimpleNamespace(_config=SimpleNamespace(batch_size=32)))
    adapter = ConnectorAdapterFactory(connector, agent).create("experiment")

    adapter.write_params({"ignored": 1})
    adapter.before_write_metric(epoch=1, step=2)
    adapter.write_metric(epoch=1, step=3, name="loss", value=0.5)
    adapter.after_write_metric(epoch=1, step=3)
    adapter.save_model(epoch=4, algo=None)
    adapter.close()

    assert connector.value_sequences_to_log == {"loss": [(96, 0.5)]}
    assert connector.uploads == [{"agent": agent, "video_recording_environment": None, "checkpoint_id": 4}]

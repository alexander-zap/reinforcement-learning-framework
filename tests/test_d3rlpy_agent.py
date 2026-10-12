"""D3RLPYAgent: offline training on recorded episodes, connector logging/checkpoints, saving/loading and ONNX.

Strict xfails document unresolved findings; they assert the desired behavior.
"""

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
# The smallest step count which trains (see the xfail below): one logging interval of 313 batches of 32.
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


def test_trained_agent_acts_and_logs_losses_and_checkpoints(trained, observations):
    agent, connector = trained

    assert set(actions(agent, observations)) <= {0, 1}
    assert connector.value_sequences_to_log["loss"]
    assert [step for step, _ in connector.value_sequences_to_log["loss"]] == [313 * 32]
    assert [upload["checkpoint_id"] for upload in connector.uploads] == [313]


@pytest.mark.xfail(
    strict=True,
    raises=UnboundLocalError,
    reason="d3rlpy.py:315-321 passes logging_steps=ceil(10000 / batch_size) even when n_steps is smaller, "
    "and d3rlpy's fitter then reads metrics that were never logged",
)
def test_short_training_runs(demonstrations):
    train(InProcessD3RLPYAgent(DiscreteBC, {"batch_size": 32}), demonstrations, total_timesteps=320)


@pytest.mark.xfail(
    strict=True,
    raises=NotImplementedError,
    reason="d3rlpy.py:309-313 always adds a TDErrorEvaluator for validation episodes, "
    "which value-free algorithms like (Discrete)BC cannot compute",
)
def test_behavior_cloning_trains_with_validation_episodes(demonstrations):
    train(
        InProcessD3RLPYAgent(DiscreteBC, {"batch_size": 32}),
        demonstrations,
        validation_episode_sequence=demonstrations,
    )


@pytest.mark.xfail(
    strict=True,
    reason="d3rlpy.py:317-325 uses LoggingStrategy.STEPS, but d3rlpy adds evaluator scores after the epoch's last "
    "step commit, so episode_reward_mean/td_error of an epoch are logged in the next epoch (and never for the last)",
)
def test_evaluation_metrics_are_logged_to_the_connector(demonstrations):
    connector = RecordingConnector()

    train(
        InProcessD3RLPYAgent(DQN, {"batch_size": 32}),
        demonstrations,
        connector=connector,
        validation_episode_sequence=EpisodeSequence.from_episodes([demonstrations[0]]),
    )

    assert connector.value_sequences_to_log["td_error"]
    assert connector.value_sequences_to_log["episode_reward_mean"]


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


@pytest.mark.xfail(
    strict=True,
    raises=ValueError,
    reason="d3rlpy.py:302-307 passes the raw PettingZoo env (dict observations) to the ReplayBuffer and "
    "EnvironmentEvaluator; the vectorized environment built at d3rlpy.py:272-287 is never used",
)
def test_training_on_pettingzoo_environment(observations):
    trajectory = TrajectoryWithRew(
        obs=np.zeros((65, 3), np.float32),
        acts=np.zeros(64, np.int64),
        infos=None,
        terminal=True,
        rews=np.zeros(64, np.float32),
    )
    agent = InProcessD3RLPYAgent(DiscreteBC, {"batch_size": 32})

    train_on = EpisodeSequence.from_episodes([trajectory])
    agent.train(TRAINABLE_TIMESTEPS, train_on, training_environments=[ToyParallel(), ToyParallel()])

    assert agent.choose_action(BOX.sample()) in (0, 1)


def test_save_and_load_round_trip_keeps_the_policy(trained, observations, tmp_path):
    agent, _ = trained
    agent.save_to_file(tmp_path / "agent.d3")

    loaded = InProcessD3RLPYAgent(DiscreteBC, {"batch_size": 32})
    loaded.load_from_file(tmp_path / "agent.d3")

    assert actions(loaded, observations) == actions(agent, observations)


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

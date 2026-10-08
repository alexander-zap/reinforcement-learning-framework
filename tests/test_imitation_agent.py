"""ImitationAgent with every supported imitation algorithm: training, connector callbacks, saving/loading and ONNX.

Strict xfails document unresolved findings; they assert the desired behavior.
"""

import logging
from pathlib import Path

import gymnasium as gym
import numpy as np
import onnx
import pytest
import torch as th
from imitation.algorithms.adversarial.airl import AIRL
from imitation.algorithms.adversarial.gail import GAIL
from imitation.algorithms.bc import BC
from imitation.algorithms.density import DensityAlgorithm
from imitation.algorithms.sqil import SQIL
from imitation.data.types import TrajectoryWithRew
from stable_baselines3 import PPO

from rl_framework.agent.imitation import (
    IMITATION_ALGORITHM_WRAPPER_REGISTRY,
    EpisodeSequence,
)
from rl_framework.agent.imitation.imitation.algorithms import BCAlgorithmWrapper
from tests.toys import (
    BOX,
    InProcessImitationAgent,
    Linear,
    RecordingConnector,
    ToyParallel,
)

CARTPOLE_ROLLOUT = Path(__file__).parents[1] / "data" / "cartpole_rollout"
RL_ALGO_KWARGS = {"n_steps": 64, "batch_size": 64, "n_epochs": 1, "device": "cpu"}
ALGORITHM_PARAMETERS = {
    BC: {"batch_size": 32},
    GAIL: {"demo_batch_size": 64, "gen_replay_buffer_capacity": 64, "rl_algo_kwargs": RL_ALGO_KWARGS},
    AIRL: {"demo_batch_size": 64, "gen_replay_buffer_capacity": 64, "rl_algo_kwargs": RL_ALGO_KWARGS},
    DensityAlgorithm: {"rl_algo_kwargs": RL_ALGO_KWARGS},
    SQIL: {"rl_algo_type": "DQN", "policy_type": "DQNPolicy", "rl_algo_kwargs": {"learning_starts": 10}},
}


@pytest.fixture(scope="module")
def demonstrations():
    recorded = EpisodeSequence.from_dataset(str(CARTPOLE_ROLLOUT))
    return EpisodeSequence.from_episodes([recorded[index] for index in range(3)])


@pytest.fixture(scope="module")
def observations(demonstrations):
    return np.asarray(demonstrations[0].obs[:20], dtype=np.float32)


def make_agent(algorithm_class=BC, features_extractor=None, **parameters):
    return InProcessImitationAgent(
        algorithm_class, {**ALGORITHM_PARAMETERS[algorithm_class], **parameters}, features_extractor
    )


def train(agent, demonstrations, total_timesteps=256, **kwargs):
    agent.train(total_timesteps, demonstrations, training_environments=[gym.make("CartPole-v1")], **kwargs)
    return agent


def actions(agent, observations):
    return [agent.choose_action(observation, deterministic=True) for observation in observations]


# --- construction ---


def test_every_imitation_algorithm_has_a_wrapper():
    assert set(IMITATION_ALGORITHM_WRAPPER_REGISTRY) == {BC, GAIL, AIRL, DensityAlgorithm, SQIL}
    assert isinstance(make_agent(BC).algorithm_wrapper, BCAlgorithmWrapper)


def test_variable_horizon_is_allowed_by_default_and_rl_parameters_are_consumed():
    agent = make_agent(GAIL)

    assert agent.algorithm_parameters["allow_variable_horizon"] is True
    wrapper = agent.algorithm_wrapper
    assert wrapper.rl_algo_class is PPO
    assert wrapper.rl_algo_kwargs == RL_ALGO_KWARGS
    assert "rl_algo_kwargs" not in wrapper.algorithm_parameters


def test_untrained_agent_cannot_act_save_or_export(tmp_path, observations):
    agent = make_agent()
    with pytest.raises(ValueError, match="uninitialized agent"):
        agent.choose_action(observations[0])
    with pytest.raises(AttributeError, match="non-initialized"):
        agent.save_to_file(tmp_path / "agent.zip")
    with pytest.raises(ValueError, match="uninitialized agent"):
        agent.save_policy_as_onnx(tmp_path / "agent.onnx")


def test_training_requires_demonstrations():
    with pytest.raises(ValueError, match="No transitions"):
        make_agent().train(32, EpisodeSequence(), training_environments=[gym.make("CartPole-v1")])


def test_training_rejects_unsupported_environment_types(demonstrations):
    with pytest.raises(TypeError, match="unsupported type"):
        make_agent().train(32, demonstrations, training_environments=["not an environment"])


# --- training ---


@pytest.mark.parametrize("algorithm_class", [BC, GAIL, AIRL, DensityAlgorithm, SQIL])
def test_every_algorithm_trains_and_acts(demonstrations, observations, algorithm_class):
    agent = train(make_agent(algorithm_class), demonstrations)

    assert all(action in (0, 1) for action in actions(agent, observations))


def test_bc_logs_training_and_validation_metrics_to_the_connector(demonstrations):
    connector = RecordingConnector()
    agent = make_agent(log_interval=1, rollout_interval=2, rollout_episodes=1)

    train(
        agent, demonstrations, total_timesteps=32 * 4, connector=connector, validation_episode_sequence=demonstrations
    )

    logged = connector.value_sequences_to_log
    assert len(logged["training/loss"]) == 4
    assert len(logged["validation/loss"]) == 4
    assert [step for step, _ in logged["rollout/return_mean"]] == [64, 128]


@pytest.mark.xfail(
    strict=True,
    raises=StopIteration,
    reason="bc.py:89-91,127 draws validation batches from one iterator over the validation data, "
    "which runs out when training logs more often than there are validation batches",
)
def test_bc_validation_does_not_run_out_of_batches(demonstrations):
    one_episode = EpisodeSequence.from_episodes([demonstrations[0]])  # 500 transitions = 15 batches of 32
    agent = make_agent(log_interval=1)

    train(agent, demonstrations, total_timesteps=32 * 20, validation_episode_sequence=one_episode)


@pytest.mark.parametrize("algorithm_class", [GAIL, AIRL, DensityAlgorithm, SQIL])
def test_environment_interacting_algorithms_log_episodes_to_the_connector(demonstrations, algorithm_class):
    connector = RecordingConnector()

    train(make_agent(algorithm_class), demonstrations, total_timesteps=256, connector=connector)

    assert connector.value_sequences_to_log["Episode reward"]


@pytest.mark.parametrize("algorithm_class", [GAIL, DensityAlgorithm])
def test_training_again_logs_only_to_the_new_connector(demonstrations, algorithm_class):
    first, second = RecordingConnector(), RecordingConnector()
    agent = train(make_agent(algorithm_class), demonstrations, total_timesteps=256, connector=first)
    logged_by_first = len(first.value_sequences_to_log["Episode reward"])

    train(agent, demonstrations, total_timesteps=256, connector=second)

    assert len(first.value_sequences_to_log["Episode reward"]) == logged_by_first
    assert second.value_sequences_to_log["Episode reward"]


def test_training_on_pettingzoo_environment_shares_the_policy_between_agents():
    trajectory = TrajectoryWithRew(
        obs=np.zeros((65, 3), np.float32),
        acts=np.zeros(64, np.int64),
        infos=None,
        terminal=True,
        rews=np.zeros(64, np.float32),
    )
    agent = make_agent()

    agent.train(64, EpisodeSequence.from_episodes([trajectory]), training_environments=[ToyParallel()])

    assert agent.choose_action(BOX.sample()) in (0, 1)


def test_training_again_reuses_the_algorithm_with_new_demonstrations(demonstrations):
    agent = train(make_agent(), demonstrations, total_timesteps=32)
    algorithm = agent.algorithm

    train(agent, EpisodeSequence.from_episodes([demonstrations[0]]), total_timesteps=32)

    assert agent.algorithm is algorithm


def test_training_on_several_pettingzoo_environments_warns_and_uses_the_first(caplog):
    trajectory = TrajectoryWithRew(
        obs=np.zeros((65, 3), np.float32),
        acts=np.zeros(64, np.int64),
        infos=None,
        terminal=True,
        rews=np.zeros(64, np.float32),
    )

    with caplog.at_level(logging.WARNING):
        make_agent().train(
            64, EpisodeSequence.from_episodes([trajectory]), training_environments=[ToyParallel(), ToyParallel()]
        )

    assert "does not support training on multiple multi-agent environments" in caplog.text


def test_loading_a_save_of_another_algorithm_loads_only_the_policy(demonstrations, observations, tmp_path, caplog):
    agent = train(make_agent(BC), demonstrations)
    agent.save_to_file(tmp_path / "agent.zip")

    loaded = make_agent(DensityAlgorithm)
    with caplog.at_level(logging.WARNING):
        loaded.load_from_file(tmp_path / "agent.zip")

    assert "Only the policy will be loaded" in caplog.text
    assert actions(loaded, observations) == actions(agent, observations)


@pytest.mark.parametrize("algorithm_class", [GAIL, AIRL, DensityAlgorithm])
def test_environment_interacting_algorithms_train_with_features_extractor(demonstrations, algorithm_class):
    agent = train(make_agent(algorithm_class, features_extractor=Linear(input_dim=4)), demonstrations)
    assert agent.algorithm_policy.features_extractor.features_dim == Linear.output_dim


def test_bc_trains_with_features_extractor(demonstrations, observations, tmp_path):
    agent = train(make_agent(features_extractor=Linear(input_dim=4)), demonstrations)
    agent.save_to_file(tmp_path / "agent.zip")

    loaded = make_agent(features_extractor=Linear(input_dim=4))
    loaded.load_from_file(tmp_path / "agent.zip")

    assert actions(loaded, observations) == actions(agent, observations)
    assert loaded.algorithm_policy.features_extractor.features_dim == Linear.output_dim


# --- saving, loading and ONNX ---


def test_bc_save_and_load_round_trip_continues_training(demonstrations, observations, tmp_path):
    agent = train(make_agent(), demonstrations)
    agent.save_to_file(tmp_path / "agent.zip")
    assert (tmp_path / "agent.zip").exists()

    loaded = make_agent()
    loaded.load_from_file(tmp_path / "agent.zip")
    assert actions(loaded, observations) == actions(agent, observations)

    with pytest.raises(AttributeError, match="loaded model without re-training"):
        loaded.save_to_file(tmp_path / "again.zip")

    train(loaded, demonstrations, total_timesteps=32)
    loaded.save_to_file(tmp_path / "again.zip")


@pytest.mark.parametrize("algorithm_class", [DensityAlgorithm, SQIL])
def test_save_and_load_round_trip_keeps_the_policy(demonstrations, observations, tmp_path, algorithm_class):
    agent = train(make_agent(algorithm_class), demonstrations)
    agent.save_to_file(tmp_path / "agent.zip")

    loaded = make_agent(algorithm_class)
    loaded.load_from_file(tmp_path / "agent.zip")

    assert actions(loaded, observations) == actions(agent, observations)
    train(loaded, demonstrations, total_timesteps=64)


@pytest.mark.parametrize("algorithm_class", [GAIL, AIRL])
def test_adversarial_save_and_load_round_trip_keeps_the_policy_and_reward_net(
    demonstrations, observations, tmp_path, algorithm_class
):
    agent = train(make_agent(algorithm_class), demonstrations)
    agent.save_to_file(tmp_path / "agent.zip")

    loaded = make_agent(algorithm_class)
    loaded.load_from_file(tmp_path / "agent.zip")

    assert actions(loaded, observations) == actions(agent, observations)
    saved_reward_net = agent.algorithm._reward_net.state_dict()
    loaded_reward_net = loaded.algorithm_wrapper.loaded_parameters["reward_net"].state_dict()
    assert all(th.equal(saved_reward_net[name], loaded_reward_net[name]) for name in saved_reward_net)
    train(loaded, demonstrations, total_timesteps=64)


def test_load_with_new_parameters_updates_them(demonstrations, tmp_path):
    train(make_agent(), demonstrations).save_to_file(tmp_path / "agent.zip")

    loaded = make_agent()
    loaded.load_from_file(tmp_path / "agent.zip", algorithm_parameters={"batch_size": 16})

    assert loaded.algorithm_parameters["batch_size"] == 16


def test_policy_exports_to_a_valid_onnx_model(demonstrations, tmp_path):
    agent = train(make_agent(), demonstrations)

    agent.save_policy_as_onnx(tmp_path / "policy.onnx")

    onnx.checker.check_model(onnx.load(tmp_path / "policy.onnx"))
    with pytest.raises(AssertionError, match=".onnx"):
        agent.save_policy_as_onnx(tmp_path / "policy.zip")

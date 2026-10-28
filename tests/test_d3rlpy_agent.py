"""D3RLPYAgent: offline training on recorded episodes, connector logging/checkpoints, saving/loading and ONNX."""

from pathlib import Path
from types import SimpleNamespace

import d3rlpy
import gymnasium as gym
import numpy as np
import onnx
import onnxruntime
import pytest
import torch as th
from d3rlpy.algos import (
    DQN,
    TD3,
    DiscreteBC,
    DiscreteDecisionTransformer,
    GreedyTransformerActionSampler,
)
from imitation.data.types import TrajectoryWithRew
from stable_baselines3.common.vec_env import DummyVecEnv

from rl_framework.agent.imitation import EpisodeSequence
from rl_framework.agent.imitation.d3rlpy.d3rlpy import (
    D3RLPY_ALGORITHM_CONFIG_REGISTRY,
    ConnectorAdapterFactory,
)
from rl_framework.util import FeaturesExtractor, record_video, video_recording
from tests.toys import (
    BOX,
    InProcessD3RLPYAgent,
    Linear,
    RecordingConnector,
    Toy,
    ToyParallel,
)

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


def train_decision_transformer(demonstrations, parameters, connector):
    return train(
        InProcessD3RLPYAgent(DiscreteDecisionTransformer, {"batch_size": 32, "context_size": 5, **parameters}),
        demonstrations,
        total_timesteps=320,
        connector=connector,
        validation_episode_sequence=EpisodeSequence.from_episodes([demonstrations[0]]),
    )


def test_decision_transformers_are_evaluated_online_with_a_target_return_and_export_to_onnx(
    demonstrations, caplog, tmp_path
):
    connector = RecordingConnector()

    agent = train_decision_transformer(demonstrations, {"target_return": 500.0}, connector)
    agent.save_policy_as_onnx(tmp_path / "policy.onnx")

    assert logged_steps(connector, "loss") == [320]
    assert logged_steps(connector, "episode_reward_mean") == [320]
    # d3rlpy's validation evaluators choose actions from single observations
    assert set(connector.value_sequences_to_log) & {"action_match", "td_error"} == set()
    assert "not evaluated on validation episodes" in caplog.text
    assert [upload["checkpoint_id"] for upload in connector.uploads] == [10]
    onnx.checker.check_model(onnx.load(tmp_path / "policy.onnx"))


def test_decision_transformers_without_a_target_return_train_without_online_evaluation(demonstrations, caplog):
    connector = RecordingConnector()

    train_decision_transformer(demonstrations, {}, connector)

    assert logged_steps(connector, "loss") == [320]
    assert "episode_reward_mean" not in connector.value_sequences_to_log
    assert "without the parameter target_return are not evaluated online" in caplog.text


def built_decision_transformer(parameters):
    agent = InProcessD3RLPYAgent(DiscreteDecisionTransformer, {"batch_size": 32, "context_size": 3, **parameters})
    agent.algorithm.build_with_env(gym.make("CartPole-v1"))
    return agent


def test_decision_transformers_choose_actions_like_a_stateful_wrapper_per_episode(observations):
    agent = built_decision_transformer({"target_return": 100.0})
    wrappers = {
        key: agent.algorithm.as_stateful_wrapper(100.0, action_sampler=GreedyTransformerActionSampler())
        for key in ("a", "b")
    }
    rewards = np.linspace(0.0, 1.0, len(observations))

    # Two episodes interleaved, episode "b" restarting halfway
    for step, observation in enumerate(observations):
        for key in ("a", "b"):
            episode_start = step == 0 or (key == "b" and step == len(observations) // 2)
            if episode_start:
                wrappers[key].reset()
            reward = 0.0 if episode_start else float(rewards[step])
            action = agent.choose_action(
                observation, deterministic=True, reward=reward, episode_start=episode_start, episode_key=key
            )
            assert action == wrappers[key].predict(observation, reward)


@pytest.mark.parametrize("sb3_replay", [False, True], ids=["simple", "sb3"])
def test_decision_transformers_can_be_recorded(tmp_path, monkeypatch, sb3_replay):
    monkeypatch.setattr(video_recording.subprocess, "run", lambda command, **kwargs: SimpleNamespace(returncode=0))
    agent = InProcessD3RLPYAgent(
        DiscreteDecisionTransformer, {"batch_size": 32, "context_size": 3, "target_return": 1.0}
    )
    agent.algorithm.build_with_env(Toy(BOX, gym.spaces.Discrete(2)))

    record_video(
        agent,
        Toy(BOX, gym.spaces.Discrete(2), episode_length=3),
        tmp_path / "replay.gif",
        video_length=5,
        sb3_replay=sb3_replay,
    )


def test_decision_transformers_cannot_choose_actions_without_the_episode_context(observations):
    with pytest.raises(NotImplementedError, match="episode_key"):
        built_decision_transformer({"target_return": 100.0}).choose_action(observations[0])


def test_decision_transformers_need_a_target_return_to_choose_actions(observations):
    with pytest.raises(ValueError, match="target_return"):
        built_decision_transformer({}).choose_action(observations[0], episode_start=True, episode_key=0)


def test_target_return_is_kept_in_the_parameters_but_not_passed_to_d3rlpy(tmp_path):
    agent = built_decision_transformer({"target_return": 100.0, "device": "cpu"})
    agent.save_to_file(tmp_path / "agent.d3")

    loaded = InProcessD3RLPYAgent(
        DiscreteDecisionTransformer, {"batch_size": 32, "context_size": 3, "target_return": 50.0}
    )
    loaded.load_from_file(tmp_path / "agent.d3")

    assert agent.algorithm_parameters == {"batch_size": 32, "context_size": 3, "target_return": 100.0, "device": "cpu"}
    assert loaded.algorithm_parameters["target_return"] == 50.0


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
    """Preprocessing-only features extractor (its `forward`, in front of d3rlpy's encoders, is the identity)."""

    output_dim = 4

    def preprocess(self, observations):
        return np.asarray(observations, dtype=np.float32) * 0.5

    def forward(self, observations):
        return observations


def test_features_extractor_preprocesses_demonstrations(demonstrations, observations):
    agent = train(InProcessD3RLPYAgent(DiscreteBC, {"batch_size": 32}, Scaling()), demonstrations)
    assert set(actions(agent, observations)) <= {0, 1}


def network_features_extractor(agent):
    """The features extractor copy in the (DiscreteBC) agent's network."""
    return agent.algorithm.impl.modules.imitator._encoder.features_extractor


def test_features_extractor_is_trained_in_front_of_the_default_encoder(demonstrations):
    features_extractor = Linear(input_dim=4)
    initial_weights = features_extractor.layer.weight.detach().clone()
    agent = train(InProcessD3RLPYAgent(DiscreteBC, {"batch_size": 32}, features_extractor), demonstrations)

    encoder = agent.algorithm.impl.modules.imitator._encoder
    assert encoder.features_extractor is not features_extractor
    assert not th.equal(encoder.features_extractor.layer.weight, initial_weights)
    # The agent's own features extractor is not trained (the network has a copy)
    assert th.equal(features_extractor.layer.weight, initial_weights)
    # d3rlpy's default encoder (256x256 MLP) after the features extractor
    assert [layer.out_features for layer in encoder.encoder._layers if isinstance(layer, th.nn.Linear)] == [256, 256]
    assert agent.algorithm_parameters == {"batch_size": 32}


@pytest.mark.parametrize("hidden_units", [[], [8]])
def test_encoder_factory_defines_the_network_after_the_features_extractor(hidden_units):
    parameters = {"batch_size": 32, "encoder_factory": d3rlpy.models.VectorEncoderFactory(hidden_units=hidden_units)}
    agent = InProcessD3RLPYAgent(DiscreteBC, parameters, Linear(input_dim=4))
    agent.algorithm.build_with_env(gym.make("CartPole-v1"))

    encoder = agent.algorithm.impl.modules.imitator._encoder
    assert isinstance(encoder.features_extractor, Linear)
    assert [layer.out_features for layer in encoder.encoder._layers if isinstance(layer, th.nn.Linear)] == hidden_units


def test_every_network_has_its_own_features_extractor_copy():
    agent = InProcessD3RLPYAgent(DQN, {"batch_size": 32}, Linear(input_dim=4))
    agent.algorithm.build_with_env(gym.make("CartPole-v1"))

    modules = agent.algorithm.impl.modules
    q_function_extractor = modules.q_funcs[0]._encoder.features_extractor
    assert q_function_extractor is not modules.targ_q_funcs[0]._encoder.features_extractor


def test_continuous_critics_put_the_features_extractor_in_front_of_the_state_action_encoder(tmp_path):
    action_space = gym.spaces.Box(-1, 1, (2,), np.float32)
    agent = InProcessD3RLPYAgent(TD3, {"batch_size": 32}, Linear(input_dim=3))
    agent.algorithm.build_with_env(Toy(BOX, action_space))
    agent.action_space = action_space
    box_observations = np.random.default_rng(0).uniform(-1, 1, (5, 3)).astype(np.float32)

    modules = agent.algorithm.impl.modules
    assert isinstance(modules.policy._encoder.features_extractor, Linear)
    critic_encoder = modules.q_funcs[0]._encoder
    assert isinstance(critic_encoder.features_extractor, Linear)
    # State-action encoder: extracted features (4) concatenated with the action (2)
    assert critic_encoder.encoder._layers[0].in_features == Linear.output_dim + 2

    agent.save_to_file(tmp_path / "agent.d3")
    loaded = InProcessD3RLPYAgent(TD3, {"batch_size": 32}, Linear(input_dim=3))
    loaded.load_from_file(tmp_path / "agent.d3")
    chosen_actions = np.stack([agent.choose_action(observation) for observation in box_observations])
    np.testing.assert_allclose(
        np.stack([loaded.choose_action(observation) for observation in box_observations]), chosen_actions, rtol=1e-6
    )

    agent.save_policy_as_onnx(tmp_path / "policy.onnx")
    session = onnxruntime.InferenceSession(str(tmp_path / "policy.onnx"), providers=["CPUExecutionProvider"])
    (onnx_actions,) = session.run(["action"], {"obs": box_observations})
    np.testing.assert_allclose(onnx_actions, chosen_actions, rtol=1e-5, atol=1e-6)


@pytest.fixture(scope="module")
def trained_with_features_extractor(demonstrations):
    return train(InProcessD3RLPYAgent(DiscreteBC, {"batch_size": 32}, Linear(input_dim=4)), demonstrations)


def test_load_restores_the_trained_features_extractor(trained_with_features_extractor, observations, tmp_path):
    agent = trained_with_features_extractor
    agent.save_to_file(tmp_path / "agent.d3")

    # A new features extractor instance (other random weights): the weights come from the file
    loaded = InProcessD3RLPYAgent(DiscreteBC, {"batch_size": 32}, Linear(input_dim=4))
    loaded.load_from_file(tmp_path / "agent.d3")

    assert th.equal(network_features_extractor(loaded).layer.weight, network_features_extractor(agent).layer.weight)
    assert actions(loaded, observations) == actions(agent, observations)


class OtherLinear(Linear):
    pass


@pytest.mark.parametrize(
    "features_extractor, message",
    [
        (OtherLinear(input_dim=4), "does not match the one of the loaded model: class"),
        (Linear(input_dim=5), "does not match the one of the loaded model: shape of `layer.weight`"),
        (None, "The loaded model has a features extractor, but the agent has none"),
    ],
    ids=["other_class", "other_shapes", "none"],
)
def test_load_rejects_another_features_extractor(
    trained_with_features_extractor, tmp_path, features_extractor, message
):
    trained_with_features_extractor.save_to_file(tmp_path / "agent.d3")

    with pytest.raises(ValueError, match=message):
        InProcessD3RLPYAgent(DiscreteBC, {"batch_size": 32}, features_extractor).load_from_file(tmp_path / "agent.d3")


def test_load_rejects_a_features_extractor_if_the_model_has_none(trained, tmp_path):
    agent, _ = trained
    agent.save_to_file(tmp_path / "agent.d3")

    with pytest.raises(ValueError, match="The loaded model has no features extractor, but the agent has one"):
        InProcessD3RLPYAgent(DiscreteBC, {"batch_size": 32}, Scaling()).load_from_file(tmp_path / "agent.d3")


def test_training_on_pettingzoo_environment_uses_its_spaces_and_evaluates_online():
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
    assert logged_steps(connector, "episode_reward_mean") == [TRAINABLE_TIMESTEPS]


def test_training_on_vectorized_environment_uses_its_spaces_and_evaluates_online(demonstrations, observations):
    connector = RecordingConnector()
    environment = DummyVecEnv([lambda: gym.make("CartPole-v1"), lambda: gym.make("CartPole-v1")])

    agent = InProcessD3RLPYAgent(DiscreteBC, {"batch_size": 32})
    agent.train(TRAINABLE_TIMESTEPS, demonstrations, connector=connector, training_environments=[environment])

    assert set(actions(agent, observations)) <= {0, 1}
    assert agent.algorithm.action_size == 2
    assert logged_steps(connector, "episode_reward_mean") == [TRAINABLE_TIMESTEPS]


def test_training_on_environment_factory_uses_the_stub_spaces_without_creating_environments(
    demonstrations, observations, caplog
):
    def create_environment():
        raise AssertionError("the factory's environment must not be created")

    connector = RecordingConnector()
    factory = (gym.make("CartPole-v1"), create_environment)

    agent = InProcessD3RLPYAgent(DiscreteBC, {"batch_size": 32})
    agent.train(TRAINABLE_TIMESTEPS, demonstrations, connector=connector, training_environments=[factory])

    assert set(actions(agent, observations)) <= {0, 1}
    assert agent.algorithm.action_size == 2
    assert connector.value_sequences_to_log["loss"]
    assert "episode_reward_mean" not in connector.value_sequences_to_log
    assert "not used for online evaluation" in caplog.text


def test_online_evaluation_keeps_the_training_environment_open(demonstrations):
    closed = []
    environment = gym.make("CartPole-v1")
    environment.close = lambda: closed.append(True)
    connector = RecordingConnector()

    # Two epochs (ceil(50000 / 1000) = 50 batches each): evaluated after both on the same environment
    InProcessD3RLPYAgent(DQN, {"batch_size": 1000}).train(
        100 * 1000, demonstrations, connector=connector, training_environments=[environment]
    )

    assert logged_steps(connector, "episode_reward_mean") == [50 * 1000, 100 * 1000]
    assert closed == []


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


def test_save_and_load_keep_the_action_space_in_a_d3rlpy_file(trained, tmp_path):
    agent, _ = trained
    agent.save_to_file(tmp_path / "agent.d3")

    loaded = InProcessD3RLPYAgent(DiscreteBC, {"batch_size": 32})
    loaded.load_from_file(tmp_path / "agent.d3")

    assert agent.action_space == gym.spaces.Discrete(2)
    assert loaded.action_space == agent.action_space
    assert isinstance(d3rlpy.load_learnable(str(tmp_path / "agent.d3"), device="cpu"), DiscreteBC)


def test_files_saved_without_the_action_space_load_without_it(trained, tmp_path):
    agent, _ = trained
    agent.algorithm.save(str(tmp_path / "agent.d3"))

    loaded = InProcessD3RLPYAgent(DiscreteBC, {"batch_size": 32})
    loaded.load_from_file(tmp_path / "agent.d3")

    assert loaded.action_space is None


def built_continuous_agent(action_space):
    """TD3 agent with networks (built from the spaces, not trained) and the given action space."""
    agent = InProcessD3RLPYAgent(TD3, {"batch_size": 32})
    agent.algorithm.build_with_env(Toy(BOX, gym.spaces.Box(-1, 1, (2,), np.float32)))
    agent.action_space = action_space
    return agent


def test_continuous_actions_are_clipped_to_the_action_space_also_in_the_onnx_export(tmp_path):
    action_space = gym.spaces.Box(-0.01, 0.01, (2,), np.float32)
    agent = built_continuous_agent(action_space)
    box_observations = np.random.default_rng(0).uniform(-1, 1, (5, 3)).astype(np.float32)

    chosen_actions = np.stack([agent.choose_action(observation) for observation in box_observations])
    agent.save_policy_as_onnx(tmp_path / "policy.onnx")
    session = onnxruntime.InferenceSession(str(tmp_path / "policy.onnx"), providers=["CPUExecutionProvider"])
    (onnx_actions,) = session.run(["action"], {"obs": box_observations})

    unclipped_actions = agent.algorithm.predict(box_observations)
    assert not np.all(np.abs(unclipped_actions) <= 0.01)
    np.testing.assert_allclose(chosen_actions, np.clip(unclipped_actions, -0.01, 0.01), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(onnx_actions, chosen_actions, rtol=1e-5, atol=1e-6)


def fitted_observation_scaler(maximum=3.0):
    return d3rlpy.preprocessing.MinMaxObservationScaler(minimum=-np.ones(3), maximum=np.ones(3) * maximum)


@pytest.mark.parametrize(
    "load_parameters",
    [{}, {"observation_scaler": d3rlpy.preprocessing.MinMaxObservationScaler()}],
    ids=["without-scalers", "with-unfitted-scalers"],
)
def test_load_keeps_the_fitted_scalers_of_the_file(tmp_path, load_parameters):
    agent = InProcessD3RLPYAgent(
        TD3,
        {
            "batch_size": 32,
            "observation_scaler": fitted_observation_scaler(),
            "action_scaler": d3rlpy.preprocessing.MinMaxActionScaler(minimum=-np.ones(2) * 2, maximum=np.ones(2) * 2),
        },
    )
    agent.algorithm.build_with_env(Toy(BOX, gym.spaces.Box(-1, 1, (2,), np.float32)))
    agent.save_to_file(tmp_path / "agent.d3")
    box_observations = np.random.default_rng(0).uniform(-1, 1, (5, 3)).astype(np.float32)

    loaded = InProcessD3RLPYAgent(TD3, {"batch_size": 32, **load_parameters})
    loaded.load_from_file(tmp_path / "agent.d3")

    np.testing.assert_array_equal(loaded.algorithm.config.observation_scaler.maximum, np.ones(3) * 3)
    np.testing.assert_array_equal(loaded.algorithm.config.action_scaler.maximum, np.ones(2) * 2)
    np.testing.assert_allclose(
        loaded.algorithm.predict(box_observations), agent.algorithm.predict(box_observations), rtol=1e-6
    )


def test_load_applies_fitted_scalers_of_the_parameters(tmp_path):
    agent = InProcessD3RLPYAgent(TD3, {"batch_size": 32, "observation_scaler": fitted_observation_scaler()})
    agent.algorithm.build_with_env(Toy(BOX, gym.spaces.Box(-1, 1, (2,), np.float32)))
    agent.save_to_file(tmp_path / "agent.d3")

    loaded = InProcessD3RLPYAgent(TD3, {"batch_size": 32, "observation_scaler": fitted_observation_scaler(5.0)})
    loaded.load_from_file(tmp_path / "agent.d3")

    np.testing.assert_array_equal(loaded.algorithm.config.observation_scaler.maximum, np.ones(3) * 5)


def test_onnx_export_without_the_action_space_warns_that_continuous_actions_are_not_clipped(tmp_path, caplog):
    agent = built_continuous_agent(None)

    agent.save_policy_as_onnx(tmp_path / "policy.onnx")

    assert "not clipped" in caplog.text


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

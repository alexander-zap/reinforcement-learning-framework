"""StableBaselinesAgent: parameter handling, training on every environment type, actions, saving/loading and ONNX.

Strict xfails document unresolved findings; they assert the desired behavior.
"""

import os

import numpy as np
import onnx
import pytest
import torch as th
from gymnasium import spaces
from stable_baselines3 import DQN, PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv

from rl_framework.util import (
    GammaScheduleCallback,
    LoggingCallback,
    ResetInfoCallback,
    SavingCallback,
)
from tests.toys import BOX, Linear, RecordingConnector, SB3Agent, Toy, ToyParallel

PPO_PARAMETERS = {"n_steps": 16, "batch_size": 16, "n_epochs": 1, "device": "cpu"}


def make_agent(algorithm_class=PPO, features_extractor=None, **parameters):
    return SB3Agent(algorithm_class, {**PPO_PARAMETERS, **parameters}, features_extractor)


def policy_weights(agent):
    return {name: value.clone() for name, value in agent.algorithm.policy.state_dict().items()}


# --- construction ---


def test_required_default_parameters_are_added():
    agent = SB3Agent()

    assert agent.algorithm_class is PPO
    assert agent.algorithm_parameters["policy"] == "MlpPolicy"
    assert os.path.isdir(agent.algorithm_parameters["tensorboard_log"])
    assert agent.algorithm_needs_initialization is True


def test_framework_parameters_are_consumed_and_user_parameters_left_unchanged():
    parameters = {"policy": "MlpPolicy", "callback_kwargs": {"callback_logging_interval": 3}, "reset_optimizer": True}

    agent = SB3Agent(PPO, parameters)

    assert agent.callback_parameters == {"callback_logging_interval": 3}
    assert agent.reset_optimizer is True
    assert "callback_kwargs" not in agent.algorithm_parameters
    assert "reset_optimizer" not in agent.algorithm_parameters
    assert parameters == {
        "policy": "MlpPolicy",
        "callback_kwargs": {"callback_logging_interval": 3},
        "reset_optimizer": True,
    }


def test_default_callbacks_follow_callback_kwargs():
    agent = SB3Agent(
        PPO,
        {
            "callback_kwargs": {
                "callback_saving_interval": 7,
                "callback_logging_interval": 3,
                "callback_log_distributions": True,
            }
        },
    )
    connector = RecordingConnector()

    saving, logging_, reset_info = agent.get_callbacks(connector)

    assert isinstance(saving, SavingCallback) and saving.checkpoint_frequency == 7 and saving.agent is agent
    assert isinstance(logging_, LoggingCallback) and logging_.logging_frequency == 3 and logging_.log_distributions
    assert isinstance(reset_info, ResetInfoCallback) and reset_info.connector is connector


def test_gamma_schedule_adds_a_callback():
    agent = SB3Agent(PPO, {"gamma": lambda progress_remaining: 0.9})
    assert isinstance(agent.get_callbacks(RecordingConnector())[-1], GammaScheduleCallback)


# --- training ---


def test_train_without_environments_is_rejected():
    with pytest.raises(ValueError, match="No training environments"):
        make_agent().train(total_timesteps=16, training_environments=[])


def test_train_on_unsupported_environment_type_is_rejected():
    with pytest.raises(TypeError, match="not supported"):
        make_agent().train(total_timesteps=16, training_environments=["not an environment"])


def test_train_on_gym_environments_vectorizes_them_and_logs_episodes():
    agent = make_agent()
    connector = RecordingConnector()

    agent.train(total_timesteps=32, connector=connector, training_environments=[Toy(), Toy()])

    assert agent.algorithm.n_envs == 2
    assert agent.algorithm.num_timesteps >= 32
    assert agent.algorithm_needs_initialization is False
    assert connector.value_sequences_to_log["Episode reward"]
    assert all(reward == 5.0 for _, reward in connector.value_sequences_to_log["Episode reward"])


def test_train_on_a_vec_env_uses_it_directly():
    agent = make_agent()
    agent.train(total_timesteps=16, training_environments=[DummyVecEnv([Toy, Toy, Toy])])
    assert agent.algorithm.n_envs == 3


def test_train_on_environment_factories_passes_the_stub_environment():
    received = {}

    class RecordingAgent(SB3Agent):
        def to_vectorized_env(self, env_fns, stub_env=None):
            received["stub_env"] = stub_env
            received["n_env_fns"] = len(env_fns)
            return super().to_vectorized_env(env_fns, stub_env)

    stub = Toy()
    agent = RecordingAgent(PPO, PPO_PARAMETERS)
    agent.train(total_timesteps=16, training_environments=[(stub, Toy), (stub, Toy)])

    assert received == {"stub_env": stub, "n_env_fns": 2}
    assert agent.algorithm.n_envs == 2


def test_train_on_pettingzoo_environment_treats_each_agent_as_an_env():
    agent = make_agent()
    connector = RecordingConnector()

    agent.train(total_timesteps=32, connector=connector, training_environments=[ToyParallel()])

    assert agent.algorithm.n_envs == 2
    assert agent.algorithm.num_timesteps >= 32
    assert connector.value_sequences_to_log["Episode reward"]


class CountingParallel(ToyParallel):
    """The reward of the t-th step of an episode is t, so every 5-step episode has a return of 1+2+3+4+5 = 15."""

    def step(self, actions):
        observations, rewards, terminations, truncations, infos = super().step(actions)
        return observations, {agent: float(self.t) for agent in rewards}, terminations, truncations, infos


@pytest.mark.xfail(
    strict=True,
    reason="stable_baselines.py:163-196 AutoResetSB3VecEnvWrapper reports each done one step late with repeated "
    "rewards: the step before done is counted twice and the first reward of the next episode is replaced, "
    "so returns of 15 are trained and logged as 19, then 18",
)
def test_pettingzoo_training_sees_the_true_episode_returns():
    agent = SB3Agent(PPO, {"n_steps": 64, "batch_size": 64, "n_epochs": 1, "device": "cpu"})
    connector = RecordingConnector()

    agent.train(total_timesteps=64, connector=connector, training_environments=[CountingParallel()])

    returns = [reward for _, reward in connector.value_sequences_to_log["Episode reward"]]
    assert returns
    assert set(returns) == {15.0}
    assert {episode["l"] for episode in agent.algorithm.ep_info_buffer} == {5}


def test_saving_interval_uploads_checkpoints_during_training():
    agent = make_agent(callback_kwargs={"callback_saving_interval": 10})
    connector = RecordingConnector()

    agent.train(total_timesteps=32, connector=connector, training_environments=[Toy()])

    checkpoints = [upload["checkpoint_id"] for upload in connector.uploads]
    assert checkpoints == [11, 22]
    assert all(upload["agent"] is agent for upload in connector.uploads)


def test_retraining_continues_from_the_trained_weights(monkeypatch):
    agent = make_agent()
    agent.train(total_timesteps=16, training_environments=[Toy()])
    trained = policy_weights(agent)

    weights_at_second_start = {}
    original_learn = PPO.learn

    def learn(algorithm, *args, **kwargs):
        weights_at_second_start.update(policy_weights(agent))
        return original_learn(algorithm, *args, **kwargs)

    monkeypatch.setattr(PPO, "learn", learn)
    agent.train(total_timesteps=16, training_environments=[Toy()])

    assert weights_at_second_start.keys() == trained.keys()
    assert all(th.equal(trained[name], weights_at_second_start[name]) for name in trained)


def test_train_with_features_extractor_uses_it_in_the_policy():
    agent = make_agent(features_extractor=Linear())
    agent.train(total_timesteps=16, training_environments=[Toy()])

    assert agent.algorithm.policy.features_extractor.features_dim == Linear.output_dim
    assert isinstance(agent.choose_action(BOX.sample()), int)


def test_retraining_with_features_extractor_works():
    agent = make_agent(features_extractor=Linear())
    agent.train(total_timesteps=16, training_environments=[Toy()])
    agent.train(total_timesteps=16, training_environments=[Toy()])


@pytest.mark.xfail(
    strict=True,
    raises=RuntimeError,
    reason="stable_baselines.py:254 reloads with custom_objects=algorithm_parameters, whose user policy_kwargs "
    "replace the saved ones that carry the features extractor, so the saved state_dict no longer fits",
)
def test_retraining_with_features_extractor_and_policy_kwargs_works():
    agent = make_agent(features_extractor=Linear(), policy_kwargs={"net_arch": [8]})
    agent.train(total_timesteps=16, training_environments=[Toy()])
    agent.train(total_timesteps=16, training_environments=[Toy()])


def test_evaluate_trained_agent_on_gym_environment():
    agent = make_agent()
    agent.train(total_timesteps=16, training_environments=[Toy()])

    mean_reward, std_reward = agent.evaluate([Toy()], n_eval_episodes=3, deterministic=True)

    assert (mean_reward, std_reward) == (5.0, 0.0)


# --- actions ---


def test_discrete_actions_are_returned_as_python_scalars():
    agent = make_agent()
    agent.train(total_timesteps=16, training_environments=[Toy()])

    action = agent.choose_action(BOX.sample(), deterministic=True)

    assert isinstance(action, int)
    assert action in (0, 1)


def test_box_actions_are_returned_as_arrays():
    agent = make_agent()
    agent.train(total_timesteps=16, training_environments=[Toy(action_space=spaces.Box(-1, 1, (2,)))])

    action = agent.choose_action(BOX.sample())

    assert isinstance(action, np.ndarray)
    assert action.shape == (2,)


# --- saving and loading ---


def test_save_and_load_round_trip_keeps_the_policy(tmp_path):
    agent = make_agent()
    agent.train(total_timesteps=16, training_environments=[Toy()])
    agent.save_to_file(tmp_path / "agent.zip")

    loaded = SB3Agent(PPO, PPO_PARAMETERS)
    loaded.load_from_file(tmp_path / "agent.zip")

    assert loaded.algorithm_needs_initialization is False
    observations = [BOX.sample() for _ in range(20)]
    assert [agent.choose_action(o, True) for o in observations] == [loaded.choose_action(o, True) for o in observations]

    loaded.train(total_timesteps=16, training_environments=[Toy()])


def test_load_with_new_parameters_applies_them_and_a_gamma_schedule(tmp_path):
    agent = make_agent()
    agent.train(total_timesteps=16, training_environments=[Toy()])
    agent.save_to_file(tmp_path / "agent.zip")

    loaded = SB3Agent(PPO, PPO_PARAMETERS)
    loaded.load_from_file(tmp_path / "agent.zip", {"learning_rate": 0.123, "gamma": lambda progress_remaining: 0.5})

    assert loaded.algorithm.learning_rate == 0.123
    assert loaded.algorithm.gamma == 0.5
    assert loaded.gamma_schedule(1.0) == 0.5


def test_agent_download_loads_from_connector(tmp_path):
    agent = make_agent()
    agent.train(total_timesteps=16, training_environments=[Toy()])
    agent.save_to_file(tmp_path / "agent.zip")

    downloaded = SB3Agent(PPO, PPO_PARAMETERS)
    downloaded.download(RecordingConnector(download_path=tmp_path / "agent.zip"))

    assert downloaded.algorithm_needs_initialization is False


# --- ONNX export ---


def export(agent, tmp_path, environment):
    agent.train(total_timesteps=16, training_environments=[environment])
    file_path = tmp_path / "policy.onnx"
    agent.save_policy_as_onnx(file_path)
    model = onnx.load(file_path)
    onnx.checker.check_model(model)
    return model


@pytest.mark.parametrize(
    "algorithm_class, environment, parameters",
    [
        (PPO, Toy(), PPO_PARAMETERS),
        (DQN, Toy(), {"learning_starts": 8, "device": "cpu"}),
        (SAC, Toy(action_space=spaces.Box(-1, 1, (2,))), {"learning_starts": 8, "batch_size": 8, "device": "cpu"}),
    ],
)
def test_policy_exports_to_a_valid_onnx_model(tmp_path, algorithm_class, environment, parameters):
    model = export(SB3Agent(algorithm_class, parameters), tmp_path, environment)

    assert [graph_input.name for graph_input in model.graph.input] == ["input"]
    dimensions = model.graph.input[0].type.tensor_type.shape.dim
    assert [dimension.dim_value for dimension in dimensions] == [1, 3]


def test_onnx_export_requires_onnx_file_ending(tmp_path):
    with pytest.raises(AssertionError, match=".onnx"):
        make_agent().save_policy_as_onnx(tmp_path / "policy.zip")


@pytest.mark.xfail(
    strict=True,
    reason="stable_baselines.py:341 exports policy(observation, deterministic=False) (see FIXME), so the ONNX graph "
    "samples its action with a Multinomial node instead of taking the most likely action",
)
def test_onnx_export_is_deterministic(tmp_path):
    model = export(make_agent(), tmp_path, Toy())
    assert "Multinomial" not in {node.op_type for node in model.graph.node}


@pytest.mark.xfail(
    strict=True,
    raises=TypeError,
    reason="stable_baselines.py:345-347 builds the dummy input from observation_space.shape, which is None for Dict",
)
def test_onnx_export_supports_dict_observations(tmp_path):
    agent = make_agent(policy="MultiInputPolicy")
    export(agent, tmp_path, Toy(observation_space=spaces.Dict({"position": BOX})))

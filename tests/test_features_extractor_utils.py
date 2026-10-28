"""Features extractor helpers: observation encoding, gym preprocessing wrapper, SB3 adapter and policy kwargs."""

import cloudpickle
import gymnasium as gym
import numpy as np
import pytest
import torch as th
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv

from rl_framework.util import (
    DummyConnector,
    FeaturesExtractor,
    StableBaselinesFeaturesExtractor,
    encode_observations_with_features_extractor,
    get_sb3_policy_kwargs_for_features_extractor,
    wrap_environment_with_features_extractor_preprocessor,
)
from tests.toys import BOX, Linear, SB3Agent, Toy, ToyParallel

DOUBLED_SPACE = spaces.Box(-2, 2, (3,), np.float32)


class Doubling(FeaturesExtractor):
    preprocessed_observation_space = DOUBLED_SPACE
    output_dim = 3

    def preprocess(self, observations):
        return np.asarray(observations) * 2

    def forward(self, observations):
        return observations


class DoublingNetwork(Doubling):
    """Doubling preprocessing, with a (trainable) linear network."""

    def __init__(self):
        super().__init__()
        self.layer = th.nn.Linear(3, 3)

    def forward(self, observations):
        return self.layer(observations.float())


def test_features_extractor_requires_forward():
    class NoForward(FeaturesExtractor):
        pass

    with pytest.raises(TypeError):
        NoForward()


def test_default_preprocess_converts_to_numpy_array():
    preprocessed = Linear().preprocess([[1, 2, 3]])
    assert isinstance(preprocessed, np.ndarray)
    np.testing.assert_array_equal(preprocessed, [[1, 2, 3]])


def test_encode_observations_runs_forward_on_the_batch():
    extractor = Linear()
    observations = [np.ones(3, np.float32), np.zeros(3, np.float32)]

    features = encode_observations_with_features_extractor(observations, extractor)

    expected = extractor(th.as_tensor(np.array(observations))).detach().numpy()
    assert isinstance(features, np.ndarray)
    np.testing.assert_allclose(features, expected)


def test_gym_wrapper_preprocesses_reset_and_step_observations():
    environment = Toy(observation_space=spaces.Box(1, 1, (3,), np.float32))
    wrapped = wrap_environment_with_features_extractor_preprocessor(environment, Doubling())

    assert isinstance(wrapped, gym.Env)
    assert wrapped.observation_space == DOUBLED_SPACE
    observation, _ = wrapped.reset()
    np.testing.assert_array_equal(observation, [2, 2, 2])
    observation, *_ = wrapped.step(0)
    np.testing.assert_array_equal(observation, [2, 2, 2])


def test_gym_wrapper_keeps_observation_space_without_preprocessed_space():
    wrapped = wrap_environment_with_features_extractor_preprocessor(Toy(), Linear())
    assert wrapped.observation_space == BOX


def test_pettingzoo_wrapper_preprocesses_final_observations_of_self_restarting_agents():
    class FinalObservationParallel(ToyParallel):
        def step(self, actions):
            observations, rewards, terminations, truncations, infos = super().step(actions)
            infos["a"]["final_observation"] = np.full(3, 0.5, np.float32)
            return observations, rewards, terminations, truncations, infos

    wrapped = wrap_environment_with_features_extractor_preprocessor(FinalObservationParallel(), Doubling())
    wrapped.reset()

    *_, infos = wrapped.step({"a": 0, "b": 0})

    np.testing.assert_array_equal(infos["a"]["final_observation"], [1, 1, 1])
    assert "final_observation" not in infos["b"]


def test_wrapping_other_environment_types_is_rejected():
    with pytest.raises(TypeError, match="not supported"):
        wrap_environment_with_features_extractor_preprocessor("not an environment", Linear())


class EndingToy(Toy):
    """Observations of ones; the episode ends after 2 steps."""

    def __init__(self):
        super().__init__(observation_space=spaces.Box(1, 1, (3,), np.float32), episode_length=2)


def test_vec_env_wrapper_preprocesses_observations_and_terminal_observations():
    wrapped = wrap_environment_with_features_extractor_preprocessor(DummyVecEnv([EndingToy, EndingToy]), Doubling())

    assert isinstance(wrapped, VecEnv)
    assert wrapped.observation_space == DOUBLED_SPACE
    np.testing.assert_array_equal(wrapped.reset(), np.full((2, 3), 2))
    wrapped.step(np.zeros(2, dtype=np.int64))
    observations, _, dones, infos = wrapped.step(np.zeros(2, dtype=np.int64))
    assert dones.all()
    np.testing.assert_array_equal(observations, np.full((2, 3), 2))
    for info in infos:
        np.testing.assert_array_equal(info["terminal_observation"], [2, 2, 2])


def test_vec_env_wrapper_passes_dict_observations_as_one_dict_per_environment():
    batches = []

    class RecordingExtractor(Linear):
        def preprocess(self, observations):
            batches.append(observations)
            return np.zeros((len(observations), 3), np.float32)

    class DictToy(Toy):
        def __init__(self):
            super().__init__(observation_space=spaces.Dict({"x": BOX}))

    wrapped = wrap_environment_with_features_extractor_preprocessor(
        DummyVecEnv([DictToy, DictToy]), RecordingExtractor()
    )
    wrapped.reset()

    assert len(batches[0]) == 2
    assert all(isinstance(observation, dict) and set(observation) == {"x"} for observation in batches[0])


@pytest.mark.parametrize(
    "env_func, check",
    [
        (EndingToy, lambda environment: isinstance(environment, gym.Env)),
        (lambda: [EndingToy(), EndingToy()], lambda environments: all(isinstance(e, gym.Env) for e in environments)),
        (lambda: DummyVecEnv([EndingToy]), lambda environment: isinstance(environment, VecEnv)),
    ],
    ids=["environment", "list", "vec_env"],
)
def test_environment_factory_wrapper_preprocesses_the_created_environments(env_func, check):
    stub, wrapped_env_func = wrap_environment_with_features_extractor_preprocessor((EndingToy(), env_func), Doubling())

    assert stub.observation_space == DOUBLED_SPACE
    created = wrapped_env_func()
    assert check(created)
    environment = created[0] if isinstance(created, list) else created
    # First observation: of the (observation, info) pair of a gym.Env, or of the first sub-environment of a VecEnv
    np.testing.assert_array_equal(environment.reset()[0], [2, 2, 2])


def test_environment_factory_wrapper_is_picklable():
    _, wrapped_env_func = wrap_environment_with_features_extractor_preprocessor((EndingToy(), EndingToy), Doubling())

    np.testing.assert_array_equal(cloudpickle.loads(cloudpickle.dumps(wrapped_env_func))().reset()[0], [2, 2, 2])


def test_policy_kwargs_extend_but_do_not_modify_user_kwargs():
    extractor = Linear()
    user_kwargs = {"net_arch": [8]}

    policy_kwargs = get_sb3_policy_kwargs_for_features_extractor(extractor, policy_kwargs=user_kwargs)

    assert policy_kwargs == {
        "net_arch": [8],
        "features_extractor_class": StableBaselinesFeaturesExtractor,
        "features_extractor_kwargs": {"features_extractor": extractor},
        "share_features_extractor": True,
    }
    assert user_kwargs == {"net_arch": [8]}


def test_sb3_adapter_uses_a_copy_of_the_extractor_with_its_output_dim():
    extractor = Linear()
    adapter = StableBaselinesFeaturesExtractor(BOX, extractor)

    assert adapter.features_dim == Linear.output_dim
    assert adapter.features_extractor is not extractor
    observations = th.ones((2, 3))
    th.testing.assert_close(adapter(observations), extractor(observations))


class ObservationRecordingAgent(SB3Agent):
    """SB3 agent which records the observations of its rollouts."""

    def get_callbacks(self, connector):
        observations = self.observations = []

        class Recorder(BaseCallback):
            def _on_step(self):
                observations.extend(self.locals["new_obs"])
                return True

        return super().get_callbacks(connector) + [Recorder()]


@pytest.mark.parametrize(
    "environment",
    [lambda: (EndingToy(), EndingToy), lambda: DummyVecEnv([EndingToy])],
    ids=["environment_factory", "vec_env"],
)
def test_sb3_agent_trains_and_evaluates_with_features_extractor_on_factories_and_vec_envs(environment):
    agent = ObservationRecordingAgent(
        PPO, {"n_steps": 8, "batch_size": 8, "n_epochs": 1, "device": "cpu"}, features_extractor=DoublingNetwork()
    )

    agent.train(total_timesteps=8, training_environments=[environment()])
    mean_reward, _ = agent.evaluate([environment()], n_eval_episodes=2, connector=DummyConnector())

    assert agent.observations
    assert all(np.array_equal(observation, [2, 2, 2]) for observation in agent.observations)
    assert mean_reward == 2.0

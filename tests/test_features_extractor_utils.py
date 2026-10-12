"""Features extractor helpers: observation encoding, gym preprocessing wrapper, SB3 adapter and policy kwargs."""

import gymnasium as gym
import numpy as np
import pytest
import torch as th
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv

from rl_framework.util import (
    FeaturesExtractor,
    StableBaselinesFeaturesExtractor,
    encode_observations_with_features_extractor,
    get_sb3_policy_kwargs_for_features_extractor,
    wrap_environment_with_features_extractor_preprocessor,
)
from tests.toys import BOX, Linear, Toy, ToyParallel

DOUBLED_SPACE = spaces.Box(-2, 2, (3,), np.float32)


class Doubling(FeaturesExtractor):
    preprocessed_observation_space = DOUBLED_SPACE
    output_dim = 3

    def preprocess(self, observations):
        return np.asarray(observations) * 2

    def forward(self, observations):
        return observations


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
    with pytest.raises(TypeError):
        wrap_environment_with_features_extractor_preprocessor(DummyVecEnv([Toy]), Linear())


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

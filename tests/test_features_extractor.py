"""FeaturesExtractor integration: SB3 target networks, the PettingZoo preprocessing wrapper, and imitation agents."""

import functools
from pathlib import Path

import gymnasium as gym
import numpy as np
import pettingzoo
import torch as th
from gymnasium import spaces
from imitation.data.types import TrajectoryWithRew
from stable_baselines3 import DQN, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.dqn.policies import DQNPolicy
from stable_baselines3.sac.policies import SACPolicy

from rl_framework.agent.imitation.d3rlpy.d3rlpy import D3RLPYAgent
from rl_framework.agent.imitation.imitation.algorithms import SQILAlgorithmWrapper
from rl_framework.agent.imitation.imitation.imitation import ImitationAgent
from rl_framework.agent.reinforcement.stable_baselines import StableBaselinesAgent
from rl_framework.util import (
    FeaturesExtractor,
    get_sb3_policy_kwargs_for_features_extractor,
    wrap_environment_with_features_extractor_preprocessor,
)

RAW_SPACE = spaces.Box(-1, 1, (4,), np.float32)
PREPROCESSED_SPACE = spaces.Box(-2, 2, (2,), np.float32)


class Linear(FeaturesExtractor):
    output_dim = 3

    def __init__(self, input_dim=4):
        super().__init__()
        self.layer = th.nn.Linear(input_dim, 3)

    def forward(self, observations):
        return self.layer(observations)


class Halving(FeaturesExtractor):
    """Preprocesses (n, 4) observations to (n, 2) by keeping the first two columns."""

    preprocessed_observation_space = PREPROCESSED_SPACE
    output_dim = 2

    def preprocess(self, observations):
        return np.asarray(observations)[:, :2]

    def forward(self, observations):
        return observations


class ToyParallel(pettingzoo.ParallelEnv):
    metadata = {"name": "toy_parallel"}

    def __init__(self):
        self.possible_agents = ["a", "b"]
        self.agents = list(self.possible_agents)

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return RAW_SPACE

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return spaces.Discrete(2)

    def reset(self, seed=None, options=None):
        self.agents = list(self.possible_agents)
        return {agent: np.ones(4, np.float32) for agent in self.agents}, {agent: {} for agent in self.agents}

    def step(self, actions):
        observations = {agent: np.full(4, 0.5, np.float32) for agent in self.agents}
        zeros = {agent: 0.0 for agent in self.agents}
        falses = {agent: False for agent in self.agents}
        return observations, zeros, falses, dict(falses), {agent: {} for agent in self.agents}


def test_sac_target_network_does_not_share_features_extractor_weights():
    algorithm = SAC(
        "MlpPolicy",
        "Pendulum-v1",
        policy_kwargs=get_sb3_policy_kwargs_for_features_extractor(Linear(input_dim=3)),
    )
    online = algorithm.critic.features_extractor.features_extractor
    target = algorithm.critic_target.features_extractor.features_extractor
    assert online is not target
    assert all(th.equal(a, b) for a, b in zip(online.parameters(), target.parameters()))

    weights_before = [p.clone() for p in online.parameters()]
    polyak_update(algorithm.critic.parameters(), algorithm.critic_target.parameters(), 0.005)
    assert all(th.equal(a, b) for a, b in zip(weights_before, online.parameters()))


def test_dqn_policy_accepts_features_extractor_and_keeps_target_separate():
    algorithm = DQN(
        "MlpPolicy",
        "CartPole-v1",
        policy_kwargs=get_sb3_policy_kwargs_for_features_extractor(Linear(), DQN.policy_aliases["MlpPolicy"]),
    )
    online = algorithm.q_net.features_extractor.features_extractor
    assert online is not algorithm.q_net_target.features_extractor.features_extractor

    weights_before = [p.clone() for p in online.parameters()]
    polyak_update(algorithm.q_net.parameters(), algorithm.q_net_target.parameters(), 1.0)
    assert all(th.equal(a, b) for a, b in zip(weights_before, online.parameters()))


def test_sqil_with_dqn_builds_with_features_extractor():
    vectorized_environment = make_vec_env("CartPole-v1", n_envs=1)
    trajectory = TrajectoryWithRew(
        obs=np.zeros((3, 4), np.float32),
        acts=np.zeros(2, np.int64),
        infos=None,
        terminal=True,
        rews=np.zeros(2, np.float32),
    )
    wrapper = SQILAlgorithmWrapper({"rl_algo_type": "DQN", "policy_type": "DQNPolicy"}, Linear())
    algorithm = wrapper.build_algorithm([trajectory], vectorized_environment)
    assert isinstance(algorithm.rl_algo, DQN)
    assert "share_features_extractor" not in algorithm.rl_algo.policy_kwargs
    vectorized_environment.close()


def test_policy_kwargs_default_to_shared_features_extractor():
    policy_kwargs = get_sb3_policy_kwargs_for_features_extractor(Linear(), SACPolicy, {"net_arch": [8]})
    assert policy_kwargs["share_features_extractor"] is True
    assert policy_kwargs["net_arch"] == [8]
    assert "share_features_extractor" not in get_sb3_policy_kwargs_for_features_extractor(Linear(), DQNPolicy)


def test_policy_kwargs_keep_explicit_share_features_extractor():
    user_policy_kwargs = {"share_features_extractor": False}
    policy_kwargs = get_sb3_policy_kwargs_for_features_extractor(Linear(), SACPolicy, user_policy_kwargs)
    assert policy_kwargs["share_features_extractor"] is False
    assert user_policy_kwargs == {"share_features_extractor": False}


def test_sqil_with_sac_keeps_unshared_features_extractor_through_save_and_load(tmp_path: Path):
    vectorized_environment = make_vec_env("Pendulum-v1", n_envs=1)
    trajectory = TrajectoryWithRew(
        obs=np.zeros((3, 3), np.float32),
        acts=np.zeros((2, 1), np.float32),
        infos=None,
        terminal=True,
        rews=np.zeros(2, np.float32),
    )
    algorithm_parameters = {
        "rl_algo_type": "SAC",
        "policy_type": "SACPolicy",
        "policy_kwargs": {"share_features_extractor": False},
    }
    wrapper = SQILAlgorithmWrapper(algorithm_parameters, Linear(input_dim=3))
    policy = wrapper.build_algorithm([trajectory], vectorized_environment).policy
    vectorized_environment.close()
    assert policy.share_features_extractor is False
    assert policy.actor.features_extractor.features_extractor is not policy.critic.features_extractor.features_extractor

    wrapper.save_policy(policy, tmp_path)
    loaded_policy = wrapper.load_policy(tmp_path)
    assert loaded_policy.share_features_extractor is False
    for name, parameter in policy.state_dict().items():
        assert th.equal(parameter.cpu(), loaded_policy.state_dict()[name].cpu()), name


class DummyVecEnvStableBaselinesAgent(StableBaselinesAgent):
    def to_vectorized_env(self, env_fns):
        return DummyVecEnv(env_fns)


def test_stable_baselines_agent_keeps_unshared_features_extractor():
    algorithm_parameters = {
        "policy": "MlpPolicy",
        "policy_kwargs": {"share_features_extractor": False},
        "learning_starts": 8,
        "batch_size": 8,
    }
    agent = DummyVecEnvStableBaselinesAgent(SAC, algorithm_parameters, Linear(input_dim=3))
    agent.train(total_timesteps=16, training_environments=[gym.make("Pendulum-v1")])

    policy = agent.algorithm.policy
    assert policy.share_features_extractor is False
    assert policy.actor.features_extractor.features_extractor is not policy.critic.features_extractor.features_extractor


def test_pettingzoo_wrapper_preprocesses_reset_and_step_observations():
    environment = ToyParallel()
    wrapped = wrap_environment_with_features_extractor_preprocessor(environment, Halving())

    assert isinstance(wrapped, pettingzoo.ParallelEnv)
    assert wrapped.observation_space("a") == PREPROCESSED_SPACE
    assert environment.observation_space("a") == RAW_SPACE

    observations, _ = wrapped.reset()
    assert {agent: obs.shape for agent, obs in observations.items()} == {"a": (2,), "b": (2,)}

    observations, *_ = wrapped.step({"a": 0, "b": 1})
    np.testing.assert_array_equal(observations["a"], [0.5, 0.5])
    assert wrapped.agents == ["a", "b"]


def test_pettingzoo_wrapper_keeps_observation_space_without_preprocessed_space():
    wrapped = wrap_environment_with_features_extractor_preprocessor(ToyParallel(), Linear())
    assert wrapped.observation_space("a") == RAW_SPACE


def test_imitation_learning_agents_are_instantiable():
    assert not ImitationAgent.__abstractmethods__
    assert not D3RLPYAgent.__abstractmethods__
    ImitationAgent(algorithm_parameters={})

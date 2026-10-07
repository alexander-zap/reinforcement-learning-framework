import copy
import inspect
from abc import ABC, abstractmethod
from typing import Any, Optional, Type

import gymnasium as gym
import numpy
import numpy as np
import pettingzoo
import torch.nn
from pettingzoo.utils.wrappers import BaseParallelWrapper
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from rl_framework.util.types import Environment


class FeaturesExtractor(ABC, torch.nn.Module):
    preprocessed_observation_space: gym.spaces.Space = None
    output_dim: int = None

    def preprocess(self, observations) -> numpy.ndarray:
        """
        Preprocess observations before feeding them to the model.
        This method by default does nothing (just converting observations into a numpy.ndarray in case they aren't yet),
            but it can be overridden for other preprocessing procedures.

        Args:
            observations: Observations to preprocess.

        Returns:
            observations: Preprocessed observations as numpy array.

        """
        return numpy.asarray(observations)

    @abstractmethod
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


def encode_observations_with_features_extractor(
    observations: list[Any], features_extractor: FeaturesExtractor
) -> np.ndarray:
    features = features_extractor.forward(torch.as_tensor(np.array(observations))).detach().numpy()
    assert len(features) == len(observations)
    return features


def wrap_environment_with_features_extractor_preprocessor(
    environment: Environment, features_extractor: FeaturesExtractor
) -> Environment:
    class FeaturesExtractorPreprocessingGymWrapper(gym.ObservationWrapper):
        def __init__(self, env, features_extractor: FeaturesExtractor):
            super().__init__(env)
            self.features_extractor = features_extractor
            self.observation_space = (
                features_extractor.preprocessed_observation_space
                if features_extractor.preprocessed_observation_space is not None
                else env.observation_space
            )

        def observation(self, observation):
            return self.features_extractor.preprocess(np.array([observation]))[0]

    class FeaturesExtractorPreprocessingPettingzooWrapper(BaseParallelWrapper):
        def __init__(self, env, features_extractor: FeaturesExtractor):
            super().__init__(env)
            self.features_extractor = features_extractor

        def observation_space(self, agent):
            if self.features_extractor.preprocessed_observation_space is not None:
                return self.features_extractor.preprocessed_observation_space
            return self.env.observation_space(agent)

        def _preprocess(self, observations: dict) -> dict:
            return {
                agent: self.features_extractor.preprocess(np.array([observation]))[0]
                for agent, observation in observations.items()
            }

        def reset(self, seed=None, options=None):
            observations, infos = self.env.reset(seed=seed, options=options)
            return self._preprocess(observations), infos

        def step(self, actions: dict):
            observations, rewards, terminations, truncations, infos = self.env.step(actions)
            return self._preprocess(observations), rewards, terminations, truncations, infos

    if isinstance(environment, pettingzoo.ParallelEnv):
        wrapped_environment = FeaturesExtractorPreprocessingPettingzooWrapper(environment, features_extractor)
    elif isinstance(environment, gym.Env):
        wrapped_environment = FeaturesExtractorPreprocessingGymWrapper(environment, features_extractor)
    else:
        raise TypeError(
            "Environment must be either a gym.Env or pettingzoo.ParallelEnv. "
            "Other types are not supported yet for using features_extractor."
        )
    return wrapped_environment


def get_sb3_policy_kwargs_for_features_extractor(
    features_extractor: FeaturesExtractor,
    policy_class: Optional[Type[BasePolicy]] = None,
    policy_kwargs: Optional[dict] = None,
) -> dict:
    """
    Build SB3 `policy_kwargs` which make the policy use the given features extractor.

    Args:
        features_extractor: Features extractor to be used by the policy.
        policy_class: SB3 policy class the kwargs are built for. `share_features_extractor` defaults to True only if
            the policy class accepts it (e.g., `DQNPolicy` does not). If None, the policy class is assumed to accept it.
        policy_kwargs: Existing policy kwargs to extend (not modified in place). An explicitly provided
            `share_features_extractor` is kept.

    Returns:
        policy_kwargs (dict): Keyword arguments to be passed to the policy constructor.
    """
    policy_kwargs = {
        **(policy_kwargs or {}),
        "features_extractor_class": StableBaselinesFeaturesExtractor,
        "features_extractor_kwargs": {"features_extractor": features_extractor},
    }
    if policy_class is None or "share_features_extractor" in inspect.signature(policy_class.__init__).parameters:
        policy_kwargs.setdefault("share_features_extractor", True)
    return policy_kwargs


class StableBaselinesFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Space, features_extractor: FeaturesExtractor):
        super().__init__(observation_space=observation_space, features_dim=features_extractor.output_dim)
        # Each SB3 construction gets its own copy, so that target networks (DQN, SAC, TD3) do not share their weights
        #   with the online network (otherwise the in-place Polyak update corrupts the shared weights).
        self.features_extractor = copy.deepcopy(features_extractor)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.features_extractor(observations)

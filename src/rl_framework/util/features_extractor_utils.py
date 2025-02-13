from abc import ABC, abstractmethod
from typing import Any

import gymnasium as gym
import numpy
import numpy as np
import torch.nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


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
    environment: gym.Env, features_extractor: FeaturesExtractor
) -> gym.Env:
    class FeaturesExtractorPreprocessingWrapper(gym.ObservationWrapper):
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

    wrapped_environment = FeaturesExtractorPreprocessingWrapper(environment, features_extractor)
    return wrapped_environment


def get_sb3_policy_kwargs_for_features_extractor(features_extractor: FeaturesExtractor) -> dict:
    return {
        "features_extractor_class": StableBaselinesFeaturesExtractor,
        "features_extractor_kwargs": {"features_extractor": features_extractor},
        "share_features_extractor": True,
    }


class StableBaselinesFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Space, features_extractor: FeaturesExtractor):
        super().__init__(observation_space=observation_space, features_dim=features_extractor.output_dim)
        self.features_extractor = features_extractor

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.features_extractor(observations)

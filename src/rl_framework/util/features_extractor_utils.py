from typing import Any

import gymnasium as gym
import numpy as np
import torch.nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


def encode_observations_with_features_extractor(
    observations: list[Any], features_extractor: torch.nn.Module
) -> np.ndarray:
    features = features_extractor(torch.as_tensor(np.array(observations))).detach().numpy()
    assert len(features) == len(observations)
    return features


def get_sb3_policy_kwargs_for_features_extractor(features_extractor: torch.nn.Module):
    return {
        "features_extractor_class": StableBaselinesFeaturesExtractor,
        "features_extractor_kwargs": {"features_extractor": features_extractor},
    }


class StableBaselinesFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Space, features_extractor: torch.nn.Module):
        features_dim = [module for module in features_extractor.modules()][-1].out_features
        super().__init__(observation_space=observation_space, features_dim=features_dim)
        self.features_extractor = features_extractor

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.features_extractor(observations)

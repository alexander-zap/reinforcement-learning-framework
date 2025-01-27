import gymnasium as gym
import torch.nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


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

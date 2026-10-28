import copy
from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
from imitation.rewards.reward_nets import BasicRewardNet, RewardNet
from imitation.util import util
from imitation.util.networks import RunningNorm
from stable_baselines3.common.preprocessing import preprocess_obs
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv

from rl_framework.util import (
    FeaturesExtractor,
    check_saved_features_extractor_description,
    describe_features_extractor,
)


class FeaturesExtractorRewardNet(RewardNet):
    """Reward network (the discriminator of GAIL and AIRL) with a features extractor: its own copy of the features
    extractor, followed by imitation's `BasicRewardNet` on the extracted features.

    The copy is trained with the discriminator loss, independent of the policy's copy. It is saved with the reward
    network (as part of the module).
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        features_extractor: FeaturesExtractor,
        **basic_reward_net_kwargs,
    ):
        """
        Args:
            observation_space: Observation space of the (preprocessed) environment.
            action_space: Action space of the environment.
            features_extractor: Features extractor to copy in front of the reward network.
            basic_reward_net_kwargs: Keyword arguments of the `BasicRewardNet` on the extracted features.
        """
        super().__init__(observation_space, action_space)
        self.features_extractor = copy.deepcopy(features_extractor)
        features_space = gym.spaces.Box(-np.inf, np.inf, (features_extractor.output_dim,), np.float32)
        self.reward_net = BasicRewardNet(features_space, action_space, **basic_reward_net_kwargs)

    def preprocess(self, state, action, next_state, done):
        # Observations as for SB3 policies (also dicts); the features extractor itself runs in `forward`, with gradients
        state_th = preprocess_obs(obs_as_tensor(state, self.device), self.observation_space, self.normalize_images)
        next_state_th = preprocess_obs(
            obs_as_tensor(next_state, self.device), self.observation_space, self.normalize_images
        )
        action_th = preprocess_obs(
            util.safe_to_tensor(action).to(self.device), self.action_space, self.normalize_images
        )
        done_th = util.safe_to_tensor(done).to(self.device).to(torch.float32)
        return state_th, action_th, next_state_th, done_th

    def forward(self, state, action, next_state, done):
        features = self.features_extractor(state)
        # Only extracted if the reward network uses the next state (not by default)
        next_features = self.features_extractor(next_state) if self.reward_net.use_next_state else features
        return self.reward_net(features, action, next_features, done)


def make_reward_net(venv: VecEnv, features_extractor: Optional[FeaturesExtractor]) -> RewardNet:
    """The reward network of GAIL and AIRL: imitation's `BasicRewardNet`, with the features extractor in front."""
    if features_extractor:
        return FeaturesExtractorRewardNet(
            venv.observation_space, venv.action_space, features_extractor, normalize_input_layer=RunningNorm
        )
    return BasicRewardNet(venv.observation_space, venv.action_space, normalize_input_layer=RunningNorm)


def load_reward_net(file_path: Path, features_extractor: Optional[FeaturesExtractor]) -> RewardNet:
    """Loads a reward network saved as a whole module; its features extractor must match the agent's (raises a
    `ValueError` otherwise)."""
    # Saved as whole module (not only weights), which torch>=2.6 does not unpickle by default
    reward_net = torch.load(file_path, weights_only=False)
    saved_features_extractor = (
        describe_features_extractor(reward_net.features_extractor)
        if isinstance(reward_net, FeaturesExtractorRewardNet)
        else None
    )
    check_saved_features_extractor_description(saved_features_extractor, features_extractor)
    return reward_net

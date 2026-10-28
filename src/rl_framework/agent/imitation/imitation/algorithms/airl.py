import tempfile
from pathlib import Path
from typing import Optional, Sequence

import torch
from imitation.algorithms.adversarial.airl import AIRL
from imitation.data import buffer
from imitation.data.types import TrajectoryWithRew
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env import VecEnv

from rl_framework.util import (
    FeaturesExtractor,
    get_sb3_policy_kwargs_for_features_extractor,
)

from .algorithm_wrapper import (
    FILE_NAME_REWARD_NET,
    FILE_NAME_SB3_ALGORITHM,
    AlgorithmWrapper,
)
from .reward_net import load_reward_net, make_reward_net


class AIRLAlgorithmWrapper(AlgorithmWrapper):
    def __init__(self, algorithm_parameters, features_extractor: Optional[FeaturesExtractor] = None):
        super().__init__(algorithm_parameters, features_extractor)
        self.venv = None

    def build_algorithm(
        self,
        trajectories: Sequence[TrajectoryWithRew],
        vectorized_environment: VecEnv,
    ) -> AIRL:
        """
        Build the AIRL algorithm with the given parameters.

        Args:
            trajectories: Trajectories to train the imitation algorithm on.
            vectorized_environment: Vectorized environment (used to construct the reward function by predicting
                 similarity of policy rollouts and expert demonstrations with a continuously updated discriminator)
        Returns:
            AIRL: AIRL algorithm object initialized with the given parameters.

        """
        self.venv = vectorized_environment
        if self.features_extractor:
            self.policy_kwargs = get_sb3_policy_kwargs_for_features_extractor(
                self.features_extractor, self.policy_class, self.policy_kwargs
            )
        parameters = {
            "venv": vectorized_environment,
            "demo_batch_size": 1024,
            "gen_algo": self.loaded_parameters.get(
                "gen_algo",
                self.rl_algo_class(
                    env=vectorized_environment,
                    policy=self.policy_class,
                    policy_kwargs=self.policy_kwargs,
                    tensorboard_log=tempfile.mkdtemp(),
                    **self.rl_algo_kwargs,
                ),
            ),
            "reward_net": self.loaded_parameters.get(
                "reward_net", make_reward_net(vectorized_environment, self.features_extractor)
            ),
        }
        parameters.update(**self.algorithm_parameters)
        algorithm = AIRL(demonstrations=trajectories, **parameters)
        return algorithm

    def train(self, algorithm: AIRL, total_timesteps: int, callback_list: CallbackList, *args, **kwargs):
        self.set_additional_callbacks(algorithm, "gen_callback", callback_list)
        algorithm.gen_train_timesteps = min(algorithm.gen_train_timesteps, total_timesteps)
        algorithm._gen_replay_buffer = buffer.ReplayBuffer(
            algorithm.gen_train_timesteps,
            self.venv,
        )
        algorithm.train(total_timesteps=total_timesteps)

    def save_algorithm(self, algorithm: AIRL, folder_path: Path):
        algorithm.gen_algo.save(folder_path / FILE_NAME_SB3_ALGORITHM)
        torch.save(algorithm._reward_net, folder_path / FILE_NAME_REWARD_NET)

    def load_algorithm(self, folder_path: Path):
        gen_algo = self._load_rl_algo(folder_path, **self.rl_algo_kwargs)
        reward_net = load_reward_net(folder_path / FILE_NAME_REWARD_NET, self.features_extractor)
        self.loaded_parameters.update({"gen_algo": gen_algo, "reward_net": reward_net})

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import torch
from imitation.algorithms.base import DemonstrationAlgorithm
from imitation.data.types import TrajectoryWithRew
from stable_baselines3 import DQN, PPO, SAC
from stable_baselines3.common.base_class import BasePolicy
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.dqn.policies import DQNPolicy
from stable_baselines3.sac.policies import SACPolicy

from rl_framework.util import FeaturesExtractor, SizedGenerator

FILE_NAME_POLICY = "policy"
FILE_NAME_SB3_ALGORITHM = "algorithm.zip"
FILE_NAME_REWARD_NET = "reward_net"


POLICY_REGISTRY = {"ActorCriticPolicy": ActorCriticPolicy, "DQNPolicy": DQNPolicy, "SACPolicy": SACPolicy}

RL_ALGO_REGISTRY = {"DQN": DQN, "SAC": SAC, "PPO": PPO}


class AlgorithmWrapper(ABC):
    def __init__(self, algorithm_parameters: dict):
        """
        algorithm_parameters: Algorithm parameters to be passed to Density algorithm of imitation library.
                See Furthermore, the following parameters can additionally be provided for modification:
                    - rl_algo_type: Type of reinforcement learning algorithm to use.
                        Available types are defined in the RL_ALGO_REGISTRY. Default: PPO
                    - rl_algo_kwargs: Additional keyword arguments to pass to the RL algorithm constructor.
                    - policy_type: Type of policy to use for the RL algorithm.
                        Available types are defined in the POLICY_REGISTRY. Default: ActorCriticPolicy
                    - policy_kwargs: Additional keyword arguments to pass to the policy constructor.


        Initialize the algorithm wrapper with the given parameters.

        Args:
            algorithm_parameters: Algorithm parameters to be passed to the respective algorithm of imitation library.
                See following links to individual algorithm API:
                - https://imitation.readthedocs.io/en/latest/_modules/imitation/algorithms/bc.html#BC
                - https://imitation.readthedocs.io/en/latest/_modules/imitation/algorithms/adversarial/gail.html#GAIL
                - https://imitation.readthedocs.io/en/latest/_modules/imitation/algorithms/adversarial/airl.html#AIRL
                - https://imitation.readthedocs.io/en/latest/_modules/imitation/algorithms/density.html#DensityAlgorithm
                - https://imitation.readthedocs.io/en/latest/_modules/imitation/algorithms/sqil.html#SQIL
                Furthermore, the following parameters can additionally be provided for modification:
                    - rl_algo_type: Type of reinforcement learning algorithm to use.
                        Available types are defined in the RL_ALGO_REGISTRY. Default: PPO
                        This argument is non-functional for BC.
                    - rl_algo_kwargs: Additional keyword arguments to pass to the RL algorithm constructor.
                        This argument is non-functional for BC.
                    - policy_type: Type of policy to use for the RL algorithm.
                        Available types are defined in the POLICY_REGISTRY. Default: ActorCriticPolicy
                    - policy_kwargs: Additional keyword arguments to pass to the policy constructor.

        """
        self.loaded_parameters: dict = {}
        self.algorithm_parameters: dict = algorithm_parameters
        self.rl_algo_class = RL_ALGO_REGISTRY.get(self.algorithm_parameters.pop("rl_algo_type", "PPO"))
        self.rl_algo_kwargs = self.algorithm_parameters.pop("rl_algo_kwargs", {})
        self.policy_class = POLICY_REGISTRY.get(self.algorithm_parameters.pop("policy_type", "ActorCriticPolicy"))
        self.policy_kwargs = self.algorithm_parameters.pop("policy_kwargs", {})

    @abstractmethod
    def build_algorithm(
        self,
        trajectories: SizedGenerator[TrajectoryWithRew],
        vectorized_environment: VecEnv,
        features_extractor: Optional[FeaturesExtractor] = None,
    ) -> DemonstrationAlgorithm:
        raise NotImplementedError

    @abstractmethod
    def train(
        self, algorithm: DemonstrationAlgorithm, total_timesteps: int, callback_list: CallbackList, *args, **kwargs
    ):
        raise NotImplementedError

    @staticmethod
    def save_policy(policy: BasePolicy, folder_path: Path):
        torch.save(policy, folder_path / FILE_NAME_POLICY)

    @abstractmethod
    def save_algorithm(self, algorithm: DemonstrationAlgorithm, folder_path: Path):
        raise NotImplementedError

    def save_to_file(self, algorithm: DemonstrationAlgorithm, folder_path: Path):
        self.save_policy(algorithm.policy, folder_path)
        self.save_algorithm(algorithm, folder_path)

    @staticmethod
    def load_policy(folder_path: Path) -> BasePolicy:
        policy: BasePolicy = torch.load(folder_path / FILE_NAME_POLICY)
        return policy

    @abstractmethod
    def load_algorithm(self, folder_path: Path):
        raise NotImplementedError

    def load_from_file(self, folder_path: Path) -> BasePolicy:
        policy = self.load_policy(folder_path)
        try:
            self.load_algorithm(folder_path)
        except FileNotFoundError:
            logging.warning(
                "Existing algorithm could not be initialized from saved file. This can be due to using a "
                "different imitation algorithm class, or due to only saving the policy before manually. "
                "\nOnly the policy will be loaded. "
                "Subsequent training of the algorithm will be performed from scratch."
            )
        return policy

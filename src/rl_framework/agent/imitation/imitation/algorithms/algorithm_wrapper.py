import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional

import torch
from imitation.algorithms.base import DemonstrationAlgorithm
from imitation.data.types import TrajectoryWithRew
from stable_baselines3 import DQN, PPO, SAC
from stable_baselines3.common.base_class import BasePolicy
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.dqn.policies import DQNPolicy
from stable_baselines3.sac.policies import SACPolicy

from rl_framework.util import (
    FeaturesExtractor,
    SizedGenerator,
    get_sb3_policy_kwargs_for_features_extractor,
)

FILE_NAME_POLICY = "policy"
FILE_NAME_SB3_ALGORITHM = "algorithm.zip"
FILE_NAME_REWARD_NET = "reward_net"


POLICY_REGISTRY = {"ActorCriticPolicy": ActorCriticPolicy, "DQNPolicy": DQNPolicy, "SACPolicy": SACPolicy}

RL_ALGO_REGISTRY = {"DQN": DQN, "SAC": SAC, "PPO": PPO}


class AlgorithmWrapper(ABC):
    def __init__(self, algorithm_parameters: dict, features_extractor: Optional[FeaturesExtractor] = None):
        """
        Initialize the algorithm wrapper with the given parameters.

        algorithm_parameters: Algorithm parameters to be passed to Density algorithm of imitation library.
                See Furthermore, the following parameters can additionally be provided for modification:
                    - rl_algo_type: Type of reinforcement learning algorithm to use.
                        Available types are defined in the RL_ALGO_REGISTRY. Default: PPO
                    - rl_algo_kwargs: Additional keyword arguments to pass to the RL algorithm constructor.
                    - policy_type: Type of policy to use for the RL algorithm.
                        Available types are defined in the POLICY_REGISTRY. Default: ActorCriticPolicy
                    - policy_kwargs: Additional keyword arguments to pass to the policy constructor.

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
            features_extractor: When provided, specifies the observation processor to be
                used before the action/value prediction network.
        """
        self.loaded_parameters: dict = {}
        self.algorithm_parameters: dict = {}
        self.algorithm_parameters.update(**algorithm_parameters)  # Copy to avoid modifying the original dict
        (
            self.rl_algo_class,
            self.rl_algo_kwargs,
            self.policy_class,
            self.policy_kwargs,
        ) = self._read_and_remove_rl_algo_and_policy_algorithm_parameters()
        self.features_extractor = features_extractor

    @abstractmethod
    def build_algorithm(
        self,
        trajectories: SizedGenerator[TrajectoryWithRew],
        vectorized_environment: VecEnv,
    ) -> DemonstrationAlgorithm:
        raise NotImplementedError

    @abstractmethod
    def train(
        self, algorithm: DemonstrationAlgorithm, total_timesteps: int, callback_list: CallbackList, *args, **kwargs
    ):
        raise NotImplementedError

    def save_policy(self, policy: BasePolicy, folder_path: Path):
        features_extractor_from_kwargs = policy.features_extractor_kwargs.pop("features_extractor", None)
        assert features_extractor_from_kwargs == self.features_extractor, (
            "Features extractor located in policy_kwargs does not match the one in the "
            "algorithm_wrapper (but should, since policy_kwargs are created from the algorithm_wrapper attribute)"
        )

        policy.save((folder_path / FILE_NAME_POLICY).as_posix())
        if features_extractor_from_kwargs:
            policy.features_extractor_kwargs.update({"features_extractor": features_extractor_from_kwargs})

    @abstractmethod
    def save_algorithm(self, algorithm: DemonstrationAlgorithm, folder_path: Path):
        raise NotImplementedError

    def save_to_file(self, algorithm: DemonstrationAlgorithm, folder_path: Path):
        self.save_policy(algorithm.policy, folder_path)
        self.save_algorithm(algorithm, folder_path)

    def load_policy(self, folder_path: Path) -> BasePolicy:
        # Method mainly copied from BasePolicy.load, but with manual addition of features_extractor to policy_kwargs
        device = get_device("auto")
        saved_variables = torch.load(
            (folder_path / FILE_NAME_POLICY).as_posix(), map_location=device, weights_only=False
        )

        if self.features_extractor:
            saved_variables["data"].update(get_sb3_policy_kwargs_for_features_extractor(self.features_extractor))

        policy: BasePolicy = self.policy_class(**saved_variables["data"])
        policy.load_state_dict(saved_variables["state_dict"])
        assert policy.features_extractor.features_extractor == self.features_extractor
        policy.to(device)
        return policy

    @abstractmethod
    def load_algorithm(self, folder_path: Path):
        raise NotImplementedError

    def load_from_file(self, folder_path: Path, algorithm_parameters: Dict = None) -> BasePolicy:
        if algorithm_parameters:
            self.algorithm_parameters.update(**algorithm_parameters)
            (
                self.rl_algo_class,
                self.rl_algo_kwargs,
                self.policy_class,
                self.policy_kwargs,
            ) = self._read_and_remove_rl_algo_and_policy_algorithm_parameters()

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

    def _read_and_remove_rl_algo_and_policy_algorithm_parameters(self):
        rl_algo_class = RL_ALGO_REGISTRY.get(self.algorithm_parameters.pop("rl_algo_type", "PPO"))
        rl_algo_kwargs = self.algorithm_parameters.pop("rl_algo_kwargs", {})
        policy_class = POLICY_REGISTRY.get(self.algorithm_parameters.pop("policy_type", "ActorCriticPolicy"))
        policy_kwargs = self.algorithm_parameters.pop("policy_kwargs", {})
        return rl_algo_class, rl_algo_kwargs, policy_class, policy_kwargs

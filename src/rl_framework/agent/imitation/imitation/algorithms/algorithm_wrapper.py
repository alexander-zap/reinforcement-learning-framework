import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Sequence

import torch
from imitation.algorithms.base import DemonstrationAlgorithm
from imitation.data.types import TrajectoryWithRew
from stable_baselines3 import DQN, PPO, SAC
from stable_baselines3.common.base_class import BaseAlgorithm, BasePolicy
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.dqn.policies import DQNPolicy
from stable_baselines3.sac.policies import SACPolicy

from rl_framework.util import (
    FeaturesExtractor,
    StableBaselinesFeaturesExtractor,
    add_callbacks_to_callback,
    check_saved_features_extractor,
    check_specified_policy_kwargs,
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
        # Callback of the imitation library on the algorithm (see `set_additional_callbacks`), and its algorithm
        self._library_callback_owner: Optional[DemonstrationAlgorithm] = None
        self._library_callback = None
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
        trajectories: Sequence[TrajectoryWithRew],
        vectorized_environment: VecEnv,
    ) -> DemonstrationAlgorithm:
        raise NotImplementedError

    @abstractmethod
    def train(
        self, algorithm: DemonstrationAlgorithm, total_timesteps: int, callback_list: CallbackList, *args, **kwargs
    ):
        raise NotImplementedError

    def set_additional_callbacks(
        self, algorithm: DemonstrationAlgorithm, library_callback_attribute: str, callback_list: CallbackList
    ) -> None:
        """
        Set the callbacks which the algorithm runs in addition to the imitation library's own callback (e.g., in
        `gen_callback`). The library's callback is remembered per algorithm, so that training the same algorithm again
        replaces the additional callbacks of the previous training instead of accumulating them.

        Args:
            algorithm: Imitation algorithm whose callback attribute is set.
            library_callback_attribute: Name of the algorithm attribute holding the library's callback.
            callback_list: Additional callbacks (e.g., logging and saving) to use for this training.
        """
        if self._library_callback_owner is not algorithm:
            self._library_callback_owner = algorithm
            self._library_callback = getattr(algorithm, library_callback_attribute)
        setattr(algorithm, library_callback_attribute, add_callbacks_to_callback(callback_list, self._library_callback))

    def save_policy(self, policy: BasePolicy, folder_path: Path):
        features_extractor_from_kwargs = policy.features_extractor_kwargs.pop("features_extractor", None)
        assert features_extractor_from_kwargs == self.features_extractor, (
            "Features extractor located in policy_kwargs does not match the one in the "
            "algorithm_wrapper (but should, since policy_kwargs are created from the algorithm_wrapper attribute)"
        )

        # Mirrors BasePolicy.save, but also stores `share_features_extractor`, which e.g. SACPolicy does not save.
        #  Otherwise, loading would fall back to a shared features extractor.
        data = policy._get_constructor_parameters()
        if hasattr(policy, "share_features_extractor"):
            data["share_features_extractor"] = policy.share_features_extractor
        torch.save({"state_dict": policy.state_dict(), "data": data}, (folder_path / FILE_NAME_POLICY).as_posix())
        if features_extractor_from_kwargs:
            policy.features_extractor_kwargs.update({"features_extractor": features_extractor_from_kwargs})

    @abstractmethod
    def save_algorithm(self, algorithm: DemonstrationAlgorithm, folder_path: Path):
        raise NotImplementedError

    def save_to_file(self, algorithm: DemonstrationAlgorithm, folder_path: Path):
        self.save_policy(algorithm.policy, folder_path)
        self.save_algorithm(algorithm, folder_path)

    @staticmethod
    def _load_saved_policy_variables(folder_path: Path) -> Dict:
        """Variables of the saved policy (see `save_policy`): its `state_dict` and constructor parameters (`data`)."""
        return torch.load(
            (folder_path / FILE_NAME_POLICY).as_posix(), map_location=get_device("auto"), weights_only=False
        )

    def load_policy(self, folder_path: Path) -> BasePolicy:
        # Method mainly copied from BasePolicy.load, but with manual addition of features_extractor to policy_kwargs
        device = get_device("auto")
        saved_policy_variables = self._load_saved_policy_variables(folder_path)

        # The saved policy only contains the class of the features extractor (`save_policy` removes the extractor), so
        #  only its presence is checked. For BC, or if no algorithm was saved, this is the only features extractor check
        #  (otherwise, `_load_rl_algo` compares it fully).
        saved_has_features_extractor = saved_policy_variables["data"].get("features_extractor_class") is (
            StableBaselinesFeaturesExtractor
        )
        if saved_has_features_extractor != bool(self.features_extractor):
            raise ValueError(
                f"The loaded policy {'has a' if saved_has_features_extractor else 'has no'} features extractor, but "
                f"the agent {'has none' if saved_has_features_extractor else 'has one'}."
            )
        if self.features_extractor:
            saved_policy_variables["data"] = get_sb3_policy_kwargs_for_features_extractor(
                self.features_extractor, self.policy_class, saved_policy_variables["data"]
            )

        policy: BasePolicy = self.policy_class(**saved_policy_variables["data"])
        policy.load_state_dict(saved_policy_variables["state_dict"])
        policy.to(device)
        return policy

    @abstractmethod
    def load_algorithm(self, folder_path: Path):
        raise NotImplementedError

    def _load_rl_algo(self, folder_path: Path, **kwargs) -> BaseAlgorithm:
        """
        Load the saved SB3 algorithm (e.g., GAIL's `gen_algo`), which is used when training again.
        The loaded network keeps the features extractor it was trained with (restored from the save), but the
        observations fed to it are preprocessed by the agent's features extractor (its `preprocess`). Therefore, the
        agent's features extractor must match the saved one in class and configuration.

        Args:
            folder_path: Folder the algorithm was saved to.
            kwargs: Parameters passed to the SB3 `load`.
        """
        rl_algo = self.rl_algo_class.load(folder_path / FILE_NAME_SB3_ALGORITHM, **kwargs)
        check_saved_features_extractor(rl_algo.policy_kwargs, self.features_extractor)
        return rl_algo

    def load_from_file(self, folder_path: Path, algorithm_parameters: Dict = None) -> BasePolicy:
        """
        Load the policy (returned) and, if saved, the algorithm (used when training again).
        The given `algorithm_parameters` apply to the loaded algorithm, also its `rl_algo_kwargs`. Specified
        `policy_kwargs` have to match the saved ones, since the architecture of a saved policy cannot be changed.
        """
        if algorithm_parameters:
            self.algorithm_parameters.update(**algorithm_parameters)
            (
                self.rl_algo_class,
                self.rl_algo_kwargs,
                self.policy_class,
                self.policy_kwargs,
            ) = self._read_and_remove_rl_algo_and_policy_algorithm_parameters()
            if "policy_kwargs" in algorithm_parameters:
                check_specified_policy_kwargs(
                    self._load_saved_policy_variables(folder_path)["data"], self.policy_kwargs
                )

        try:
            self.load_algorithm(folder_path)
        except FileNotFoundError:
            logging.warning(
                "Existing algorithm could not be initialized from saved file. This can be due to using a "
                "different imitation algorithm class, or due to only saving the policy before manually. "
                "\nOnly the policy will be loaded. "
                "Subsequent training of the algorithm will be performed from scratch."
            )
        return self.load_policy(folder_path)

    def _read_and_remove_rl_algo_and_policy_algorithm_parameters(self):
        rl_algo_class = RL_ALGO_REGISTRY.get(self.algorithm_parameters.pop("rl_algo_type", "PPO"))
        rl_algo_kwargs = self.algorithm_parameters.pop("rl_algo_kwargs", {})
        policy_class = POLICY_REGISTRY.get(self.algorithm_parameters.pop("policy_type", "ActorCriticPolicy"))
        policy_kwargs = self.algorithm_parameters.pop("policy_kwargs", {})
        return rl_algo_class, rl_algo_kwargs, policy_class, policy_kwargs

import logging
from pathlib import Path
from typing import Optional

import gymnasium
import numpy as np
from imitation.algorithms.sqil import SQIL, SQILReplayBuffer
from imitation.data.types import TrajectoryWithRew, Transitions
from imitation.util import util
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.vec_env import VecEnv

from rl_framework.util import (
    FeaturesExtractor,
    SizedGenerator,
    create_memory_efficient_transition_batcher,
    get_sb3_policy_kwargs_for_features_extractor,
)

from .algorithm_wrapper import FILE_NAME_SB3_ALGORITHM, AlgorithmWrapper


class SQILAlgorithmWrapper(AlgorithmWrapper):
    def __init__(self, algorithm_parameters, features_extractor: Optional[FeaturesExtractor] = None):
        super().__init__(algorithm_parameters, features_extractor)

        batch_size = self.rl_algo_kwargs.get("batch_size", 256)

        def patched_set_demonstrations(self, demonstrations: SizedGenerator[TrajectoryWithRew]) -> None:
            demo_data_loader = create_memory_efficient_transition_batcher(
                demonstrations,
                batch_size,
            )
            batched_transitions_mapping = next(demo_data_loader)
            self.expert_buffer.sample = lambda x, y: ReplayBufferSamples(
                observations=util.safe_to_tensor(batched_transitions_mapping["obs"]),
                actions=util.safe_to_tensor(batched_transitions_mapping["acts"]),
                rewards=util.safe_to_tensor(np.array([[1.0]] * batch_size, dtype=np.float32)),
                next_observations=util.safe_to_tensor(batched_transitions_mapping["next_obs"]),
                dones=util.safe_to_tensor(batched_transitions_mapping["dones"]),
            )

        SQIL.set_demonstrations = lambda x, y: None
        SQILReplayBuffer.set_demonstrations = patched_set_demonstrations

    def build_algorithm(
        self,
        trajectories: SizedGenerator[TrajectoryWithRew],
        vectorized_environment: VecEnv,
    ) -> SQIL:
        """
        Build the SQIL algorithm with the given parameters.

        Args:
            trajectories: Trajectories to train the imitation algorithm on.
            vectorized_environment: Vectorized environment (used to train the RL algorithm, but half of the RL algorithm
                memory keeps filled with expert demonstrations; also all rewards from the live environment are set to
                0.0, while all rewards from the expert demonstrations are set to 1.0).

        Returns:
            SQIL: SQIL algorithm object initialized with the given parameters.

        """
        # FIXME: SQILReplayBuffer inherits from sb3.ReplayBuffer which doesn't support dict observations.
        #  Maybe it can be patched to inherit from sb3.DictReplayBuffer.
        assert not isinstance(
            vectorized_environment.observation_space, gymnasium.spaces.Dict
        ), "SQILReplayBuffer does not support Dict observation spaces."

        if self.features_extractor:
            self.policy_kwargs.update(get_sb3_policy_kwargs_for_features_extractor(self.features_extractor))

        parameters = {
            "venv": vectorized_environment,
            "policy": self.policy_class,
            "rl_algo_class": self.rl_algo_class,
            "rl_kwargs": {"policy_kwargs": self.policy_kwargs, **self.rl_algo_kwargs},
        }
        parameters.update(**self.algorithm_parameters)
        if parameters.pop("allow_variable_horizon", None) is not None:
            logging.warning("SQIL algorithm does not support passing of the parameter `allow_variable_horizon`.")

        algorithm = SQIL(demonstrations=trajectories, **parameters)
        if "rl_algo" in self.loaded_parameters:
            algorithm.rl_algo = self.loaded_parameters.get("rl_algo")
            algorithm.rl_algo.set_env(vectorized_environment)
            algorithm.rl_algo.replay_buffer.set_demonstrations(trajectories)
        return algorithm

    def train(self, algorithm: SQIL, total_timesteps: int, callback_list: CallbackList, *args, **kwargs):
        algorithm.train(total_timesteps=total_timesteps, callback=callback_list)

    def save_algorithm(self, algorithm: SQIL, folder_path: Path):
        algorithm.rl_algo.save(folder_path / FILE_NAME_SB3_ALGORITHM, exclude=["replay_buffer_kwargs"])

    def load_algorithm(self, folder_path: Path):
        rl_algo = self.rl_algo_class.load(
            folder_path / FILE_NAME_SB3_ALGORITHM,
            replay_buffer_kwargs={
                "demonstrations": Transitions(
                    obs=np.array([]),
                    next_obs=np.array([]),
                    acts=np.array([]),
                    dones=np.array([], dtype=bool),
                    infos=np.array([]),
                )
            },
        )
        self.loaded_parameters.update({"rl_algo": rl_algo})

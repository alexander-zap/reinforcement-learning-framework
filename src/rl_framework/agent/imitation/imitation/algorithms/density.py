from pathlib import Path
from typing import Optional

import numpy as np
from imitation.algorithms.density import DensityAlgorithm
from imitation.data.types import TrajectoryWithRew
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env import VecEnv

from rl_framework.util import (
    FeaturesExtractor,
    SizedGenerator,
    add_callbacks_to_callback,
    get_sb3_policy_kwargs_for_features_extractor,
)

from .algorithm_wrapper import FILE_NAME_SB3_ALGORITHM, AlgorithmWrapper


class DensityAlgorithmWrapper(AlgorithmWrapper):
    def __init__(self, algorithm_parameters):
        super().__init__(algorithm_parameters)

        def temporary_switch_off_looping_demonstrations(func):
            def wrap(*args, **kwargs):
                demonstrations: SizedGenerator = args[1]
                demonstrations_were_looping = demonstrations.looping
                demonstrations.looping = False
                func(*args, **kwargs)
                demonstrations.looping = demonstrations_were_looping

            return wrap

        DensityAlgorithm.set_demonstrations = temporary_switch_off_looping_demonstrations(
            DensityAlgorithm.set_demonstrations
        )

    def build_algorithm(
        self,
        trajectories: SizedGenerator[TrajectoryWithRew],
        vectorized_environment: VecEnv,
        features_extractor: Optional[FeaturesExtractor] = None,
    ) -> DensityAlgorithm:
        """
        Build the DensityAlgorithm algorithm with the given parameters.

        Args:
            trajectories: Trajectories to train the imitation algorithm on.
            vectorized_environment: Vectorized environment (used to train the RL algorithm, but with a replaced reward
                function based on log-likelihood of observed state-action pairs to a learned distribution of expert
                demonstrations; distribution of expert demonstrations is learned by kernel density estimation).
            features_extractor: Features extractor (preprocessing of observations to vectors, trainable).

        Returns:
            DensityAlgorithm: DensityAlgorithm algorithm object initialized with the given parameters.

        """
        if features_extractor:
            self.policy_kwargs.update(get_sb3_policy_kwargs_for_features_extractor(features_extractor))
        parameters = {
            "venv": vectorized_environment,
            "rng": np.random.default_rng(0),
            "rl_algo": self.loaded_parameters.get(
                "rl_algo",
                self.rl_algo_class(
                    env=vectorized_environment,
                    policy=self.policy_class,
                    policy_kwargs=self.policy_kwargs,
                    **self.rl_algo_kwargs,
                ),
            ),
        }
        parameters.update(**self.algorithm_parameters)
        algorithm = DensityAlgorithm(demonstrations=trajectories, **parameters)
        return algorithm

    def train(self, algorithm: DensityAlgorithm, total_timesteps: int, callback_list: CallbackList, *args, **kwargs):
        algorithm.train()
        # NOTE: All callbacks concerning reward calculation will use the density reward and not the environment reward
        add_callbacks_to_callback(callback_list, algorithm.wrapper_callback)
        algorithm.train_policy(n_timesteps=total_timesteps)

    def save_algorithm(self, algorithm: DensityAlgorithm, folder_path: Path):
        algorithm.rl_algo.save(folder_path / FILE_NAME_SB3_ALGORITHM)

    def load_algorithm(self, folder_path: Path):
        rl_algo = self.rl_algo_class.load(folder_path / FILE_NAME_SB3_ALGORITHM)
        self.loaded_parameters.update({"rl_algo": rl_algo})

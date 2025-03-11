import copy
import logging
import math
from pathlib import Path
from typing import Mapping, Optional

import numpy as np
import torch
from imitation.algorithms.base import DemonstrationAlgorithm
from imitation.algorithms.bc import BC, BCTrainingMetrics, RolloutStatsComputer
from imitation.data import types
from imitation.data.types import TrajectoryWithRew, TransitionMapping
from imitation.util import util
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env import VecEnv

from rl_framework.util import (
    FeaturesExtractor,
    LoggingCallback,
    SavingCallback,
    SizedGenerator,
    create_memory_efficient_transition_batcher,
    get_sb3_policy_kwargs_for_features_extractor,
)

from .algorithm_wrapper import AlgorithmWrapper


class BCAlgorithmWrapper(AlgorithmWrapper):
    def __init__(self, algorithm_parameters):
        super().__init__(algorithm_parameters)
        self.venv = None
        self.log_interval = 500
        self.rollout_interval = None
        self.rollout_episodes = 10

        def patched_set_demonstrations(self, demonstrations: SizedGenerator[TrajectoryWithRew]) -> None:
            self._demo_data_loader = create_memory_efficient_transition_batcher(demonstrations, self.minibatch_size)

        BC.set_demonstrations = patched_set_demonstrations

    def build_algorithm(
        self,
        trajectories: SizedGenerator[TrajectoryWithRew],
        vectorized_environment: VecEnv,
        features_extractor: Optional[FeaturesExtractor] = None,
    ) -> BC:
        """
        Build the BC algorithm with the given parameters.

        Args:
            trajectories: Trajectories to train the imitation algorithm on.
            vectorized_environment: Vectorized environment (used to extract observation and action space)
            features_extractor: Features extractor (preprocessing of observations to vectors, trainable).

        Returns:
            BC: BC algorithm object initialized with the given parameters.

        """
        self.venv = vectorized_environment
        if features_extractor:
            self.policy_kwargs.update(get_sb3_policy_kwargs_for_features_extractor(features_extractor))
        parameters = {
            "observation_space": vectorized_environment.observation_space,
            "action_space": vectorized_environment.action_space,
            "rng": np.random.default_rng(0),
            "policy": self.loaded_parameters.get(
                "policy",
                self.policy_class(
                    observation_space=self.venv.observation_space,
                    action_space=self.venv.action_space,
                    lr_schedule=lambda _: torch.finfo(torch.float32).max,
                    **self.policy_kwargs,
                ),
            ),
        }
        parameters.update(**self.algorithm_parameters)
        if parameters.pop("allow_variable_horizon", None) is not None:
            logging.warning("BC algorithm does not support passing of the parameter `allow_variable_horizon`.")
        self.log_interval = parameters.pop("log_interval", self.log_interval)
        self.rollout_interval = parameters.pop("rollout_interval", self.rollout_interval)
        self.rollout_episodes = parameters.pop("rollout_episodes", self.rollout_episodes)
        algorithm = BC(demonstrations=trajectories, **parameters)
        return algorithm

    def train(
        self,
        algorithm: BC,
        total_timesteps: int,
        callback_list: CallbackList,
        validation_trajectories: Optional[SizedGenerator[TrajectoryWithRew]] = None,
        *args,
        **kwargs,
    ):
        on_batch_end_functions = []

        validation_transitions_batcher = (
            iter(create_memory_efficient_transition_batcher(validation_trajectories))
            if validation_trajectories
            else None
        )

        for callback in callback_list.callbacks:
            if isinstance(callback, LoggingCallback):
                logging_callback = copy.copy(callback)

                # Wrapped log_batch function to additionally log values into the connector
                def log_batch_with_connector(
                    batch_num: int,
                    batch_size: int,
                    num_samples_so_far: int,
                    training_metrics: BCTrainingMetrics,
                    rollout_stats: Mapping[str, float],
                ):
                    # Call the original log_batch function
                    original_log_batch(batch_num, batch_size, num_samples_so_far, training_metrics, rollout_stats)

                    # Log the recorded values into the connector additionally
                    for k, v in training_metrics.__dict__.items():
                        if v is not None:
                            logging_callback.connector.log_value_with_timestep(
                                num_samples_so_far, float(v), f"training/{k}"
                            )

                # Replace the original `log_batch` function with the new one
                original_log_batch = algorithm._bc_logger.log_batch
                algorithm._bc_logger.log_batch = log_batch_with_connector

                compute_rollout_stats = RolloutStatsComputer(
                    self.venv,
                    self.rollout_episodes,
                )

                def log(batch_number):
                    # Use validation data to compute loss metrics and log it to connector
                    if validation_transitions_batcher is not None and batch_number % self.log_interval == 0:
                        validation_transitions: TransitionMapping = next(validation_transitions_batcher)
                        obs_tensor = types.map_maybe_dict(
                            lambda x: util.safe_to_tensor(x, device=algorithm.policy.device),
                            types.maybe_unwrap_dictobs(validation_transitions["obs"]),
                        )
                        acts = util.safe_to_tensor(validation_transitions["acts"], device=algorithm.policy.device)
                        validation_metrics = algorithm.loss_calculator(algorithm.policy, obs_tensor, acts)
                        for k, v in validation_metrics.__dict__.items():
                            if v is not None:
                                logging_callback.connector.log_value_with_timestep(
                                    algorithm.batch_size * batch_number, float(v), f"validation/{k}"
                                )

                    if self.rollout_interval and batch_number % self.rollout_interval == 0:
                        rollout_stats = compute_rollout_stats(algorithm.policy, np.random.default_rng(0))
                        for k, v in rollout_stats.items():
                            if "return" in k and "monitor" not in k and v is not None:
                                logging_callback.connector.log_value_with_timestep(
                                    algorithm.batch_size * batch_number,
                                    float(v),
                                    "rollout/" + k,
                                )

                on_batch_end_functions.append(log)

            elif isinstance(callback, SavingCallback):
                saving_callback = copy.copy(callback)

                def save(batch_number):
                    saving_callback.num_timesteps = algorithm.batch_size * batch_number
                    saving_callback._on_step()

                on_batch_end_functions.append(save)

        on_batch_end_counter = {func: 0 for func in on_batch_end_functions}

        def on_batch_end():
            for func in on_batch_end_functions:
                on_batch_end_counter[func] += 1
                func(on_batch_end_counter[func])

        algorithm.train(
            n_batches=math.ceil(total_timesteps / algorithm.batch_size),
            on_batch_end=on_batch_end,
            log_interval=self.log_interval,
        )

    def save_algorithm(self, algorithm: DemonstrationAlgorithm, folder_path: Path):
        pass  # only policy saving is required for this algorithm

    def load_algorithm(self, folder_path: Path):
        policy = self.load_policy(folder_path)
        self.loaded_parameters = {"policy": policy}

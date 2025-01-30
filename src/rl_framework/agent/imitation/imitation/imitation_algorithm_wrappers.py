import copy
import itertools
import logging
import math
import tempfile
from abc import ABC, abstractmethod
from itertools import tee
from pathlib import Path
from typing import Iterable, Mapping, Optional

import numpy as np
import torch
from imitation.algorithms.adversarial.airl import AIRL
from imitation.algorithms.adversarial.gail import GAIL
from imitation.algorithms.base import DemonstrationAlgorithm
from imitation.algorithms.bc import BC, BCTrainingMetrics, RolloutStatsComputer
from imitation.algorithms.density import DensityAlgorithm
from imitation.algorithms.sqil import SQIL
from imitation.data import rollout
from imitation.data.types import TrajectoryWithRew, Transitions
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util import util
from imitation.util.networks import RunningNorm
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.base_class import BasePolicy
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.ppo import MlpPolicy

from rl_framework.util import (
    FeaturesExtractor,
    LoggingCallback,
    SavingCallback,
    SizedGenerator,
    add_callbacks_to_callback,
    get_sb3_policy_kwargs_for_features_extractor,
)

FILE_NAME_POLICY = "policy"
FILE_NAME_SB3_ALGORITHM = "algorithm.zip"
FILE_NAME_REWARD_NET = "reward_net"


class AlgorithmWrapper(ABC):
    def __init__(self):
        self.loaded_parameters: dict = {}

    @abstractmethod
    def build_algorithm(
        self,
        algorithm_parameters: dict,
        total_timesteps: int,
        trajectories: SizedGenerator[TrajectoryWithRew],
        vectorized_environment: VecEnv,
        features_extractor: Optional[FeaturesExtractor] = None,
    ) -> DemonstrationAlgorithm:
        raise NotImplementedError

    @abstractmethod
    def train(self, algorithm: DemonstrationAlgorithm, total_timesteps: int, callback_list: CallbackList):
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


class BCAlgorithmWrapper(AlgorithmWrapper):
    def __init__(self):
        super().__init__()
        self.venv = None
        self.validation_transitions: Optional[Transitions] = None
        self.log_interval = 500
        self.rollout_interval = None
        self.rollout_episodes = 10

    def build_algorithm(
        self,
        algorithm_parameters: dict,
        total_timesteps: int,
        trajectories: SizedGenerator[TrajectoryWithRew],
        vectorized_environment: VecEnv,
        features_extractor: Optional[FeaturesExtractor] = None,
    ) -> BC:
        self.venv = vectorized_environment
        parameters = {
            "observation_space": vectorized_environment.observation_space,
            "action_space": vectorized_environment.action_space,
            "rng": np.random.default_rng(0),
            "policy": self.loaded_parameters.get(
                "policy",
                ActorCriticPolicy(
                    observation_space=self.venv.observation_space,
                    action_space=self.venv.action_space,
                    net_arch=[32, 32],
                    lr_schedule=lambda _: torch.finfo(torch.float32).max,
                    **(get_sb3_policy_kwargs_for_features_extractor(features_extractor) if features_extractor else {}),
                ),
            ),
        }
        parameters.update(**algorithm_parameters)
        if parameters.pop("allow_variable_horizon", None) is not None:
            logging.warning("BC algorithm does not support passing of the parameter `allow_variable_horizon`.")
        if (validation_fraction := parameters.pop("validation_fraction", None)) is not None:
            # Rest of the `trajectories` generator can be used for training with `n - n_validation` episodes left
            validation_trajectories = list(itertools.islice(trajectories, int(validation_fraction * len(trajectories))))
            self.validation_transitions = rollout.flatten_trajectories(validation_trajectories)
        self.log_interval = parameters.pop("log_interval", self.log_interval)
        self.rollout_interval = parameters.pop("rollout_interval", self.rollout_interval)
        self.rollout_episodes = parameters.pop("rollout_episodes", self.rollout_episodes)
        algorithm = BC(demonstrations=trajectories, **parameters)
        return algorithm

    def train(self, algorithm: BC, total_timesteps: int, callback_list: CallbackList):
        on_batch_end_functions = []

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
                    if self.validation_transitions is not None and batch_number % self.log_interval == 0:
                        obs_tensor = util.safe_to_tensor(self.validation_transitions.obs)
                        acts = util.safe_to_tensor(self.validation_transitions.acts, device=algorithm.policy.device)
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


class GAILAlgorithmWrapper(AlgorithmWrapper):
    def build_algorithm(
        self,
        algorithm_parameters: dict,
        total_timesteps: int,
        trajectories: SizedGenerator[TrajectoryWithRew],
        vectorized_environment: VecEnv,
        features_extractor: Optional[FeaturesExtractor] = None,
    ) -> GAIL:
        parameters = {
            "venv": vectorized_environment,
            "demo_batch_size": 1024,
            # FIXME: Hard-coded PPO as default trajectory generation algorithm
            "gen_algo": self.loaded_parameters.get(
                "gen_algo",
                PPO(
                    env=vectorized_environment,
                    policy=MlpPolicy,
                    policy_kwargs=get_sb3_policy_kwargs_for_features_extractor(features_extractor)
                    if features_extractor
                    else None,
                    tensorboard_log=tempfile.mkdtemp(),
                ),
            ),
            # FIXME: This probably will not work with Dict as observation_space.
            #  Might require extension of BasicRewardNet to use features_extractor as well.
            "reward_net": self.loaded_parameters.get(
                "reward_net",
                BasicRewardNet(
                    observation_space=vectorized_environment.observation_space,
                    action_space=vectorized_environment.action_space,
                    normalize_input_layer=RunningNorm,
                ),
            ),
        }
        parameters["gen_train_timesteps"] = min(
            total_timesteps, parameters.get("gen_algo").n_steps * vectorized_environment.num_envs
        )
        parameters.update(**algorithm_parameters)
        algorithm = GAIL(demonstrations=trajectories, **parameters)
        return algorithm

    def train(self, algorithm: GAIL, total_timesteps: int, callback_list: CallbackList):
        add_callbacks_to_callback(callback_list, algorithm.gen_callback)
        algorithm.train(total_timesteps=total_timesteps)

    def save_algorithm(self, algorithm: GAIL, folder_path: Path):
        algorithm.gen_algo.save(folder_path / FILE_NAME_SB3_ALGORITHM)
        torch.save(algorithm._reward_net, folder_path / FILE_NAME_REWARD_NET)

    def load_algorithm(self, folder_path: Path):
        # FIXME: Only works because gen_algo is hard-coded to PPO above
        gen_algo = PPO.load(folder_path / FILE_NAME_SB3_ALGORITHM)
        reward_net = torch.load(folder_path / FILE_NAME_REWARD_NET)
        self.loaded_parameters.update({"gen_algo": gen_algo, "reward_net": reward_net})


class AIRLAlgorithmWrapper(AlgorithmWrapper):
    def build_algorithm(
        self,
        algorithm_parameters: dict,
        total_timesteps: int,
        trajectories: SizedGenerator[TrajectoryWithRew],
        vectorized_environment: VecEnv,
        features_extractor: Optional[FeaturesExtractor] = None,
    ) -> AIRL:
        parameters = {
            "venv": vectorized_environment,
            "demo_batch_size": 1024,
            # FIXME: Hard-coded PPO as default trajectory generation algorithm
            "gen_algo": self.loaded_parameters.get(
                "gen_algo",
                PPO(
                    env=vectorized_environment,
                    policy=MlpPolicy,
                    policy_kwargs=get_sb3_policy_kwargs_for_features_extractor(features_extractor)
                    if features_extractor
                    else None,
                    tensorboard_log=tempfile.mkdtemp(),
                ),
            ),
            # FIXME: This probably will not work with Dict as observation_space.
            #  Might require extension of BasicRewardNet to use features_extractor as well.
            "reward_net": self.loaded_parameters.get(
                "reward_net",
                BasicRewardNet(
                    observation_space=vectorized_environment.observation_space,
                    action_space=vectorized_environment.action_space,
                    normalize_input_layer=RunningNorm,
                ),
            ),
        }
        parameters["gen_train_timesteps"]: min(
            total_timesteps, parameters["gen_algo"].n_steps * vectorized_environment.num_envs
        )
        parameters.update(**algorithm_parameters)
        algorithm = AIRL(demonstrations=trajectories, **parameters)
        return algorithm

    def train(self, algorithm: AIRL, total_timesteps: int, callback_list: CallbackList):
        add_callbacks_to_callback(callback_list, algorithm.gen_callback)
        algorithm.train(total_timesteps=total_timesteps)

    def save_algorithm(self, algorithm: AIRL, folder_path: Path):
        algorithm.gen_algo.save(folder_path / FILE_NAME_SB3_ALGORITHM)
        torch.save(algorithm._reward_net, folder_path / FILE_NAME_REWARD_NET)

    def load_algorithm(self, folder_path: Path):
        # FIXME: Only works because gen_algo is hard-coded to PPO above
        gen_algo = PPO.load(folder_path / FILE_NAME_SB3_ALGORITHM)
        reward_net = torch.load(folder_path / FILE_NAME_REWARD_NET)
        self.loaded_parameters.update({"gen_algo": gen_algo, "reward_net": reward_net})


class DensityAlgorithmWrapper(AlgorithmWrapper):
    def build_algorithm(
        self,
        algorithm_parameters: dict,
        total_timesteps: int,
        trajectories: SizedGenerator[TrajectoryWithRew],
        vectorized_environment: VecEnv,
        features_extractor: Optional[FeaturesExtractor] = None,
    ) -> DensityAlgorithm:
        parameters = {
            "venv": vectorized_environment,
            "rng": np.random.default_rng(0),
            # FIXME: Hard-coded PPO as default policy training algorithm
            #  (to learn from adjusted reward function)
            "rl_algo": self.loaded_parameters.get(
                "rl_algo",
                PPO(
                    env=vectorized_environment,
                    policy=ActorCriticPolicy,
                    policy_kwargs=get_sb3_policy_kwargs_for_features_extractor(features_extractor)
                    if features_extractor
                    else None,
                ),
            ),
        }
        parameters.update(**algorithm_parameters)
        algorithm = DensityAlgorithm(demonstrations=trajectories, **parameters)
        return algorithm

    def train(self, algorithm: DensityAlgorithm, total_timesteps: int, callback_list: CallbackList):
        algorithm.train()
        # NOTE: All callbacks concerning reward calculation will use the density reward and not the environment reward
        add_callbacks_to_callback(callback_list, algorithm.wrapper_callback)
        algorithm.train_policy(n_timesteps=total_timesteps)

    def save_algorithm(self, algorithm: DensityAlgorithm, folder_path: Path):
        algorithm.rl_algo.save(folder_path / FILE_NAME_SB3_ALGORITHM)

    def load_algorithm(self, folder_path: Path):
        # FIXME: Only works because rl_algo is hard-coded to PPO above
        rl_algo = PPO.load(folder_path / FILE_NAME_SB3_ALGORITHM)
        self.loaded_parameters.update({"rl_algo": rl_algo})


class SQILAlgorithmWrapper(AlgorithmWrapper):
    def build_algorithm(
        self,
        algorithm_parameters: dict,
        total_timesteps: int,
        trajectories: SizedGenerator[TrajectoryWithRew],
        vectorized_environment: VecEnv,
        features_extractor: Optional[FeaturesExtractor] = None,
    ) -> SQIL:
        parameters = {
            "venv": vectorized_environment,
            "policy": "MlpPolicy",
            # FIXME: Hard-coded DQN as default policy training algorithm
            "rl_algo_class": DQN,
            "rl_kwargs": {
                "policy_kwargs": get_sb3_policy_kwargs_for_features_extractor(features_extractor)
                if features_extractor
                else None,
            },
        }
        parameters.update(**algorithm_parameters)
        if parameters.pop("allow_variable_horizon", None) is not None:
            logging.warning("SQIL algorithm does not support passing of the parameter `allow_variable_horizon`.")

        class MockedIterableFromGenerator(Iterable):
            def __init__(self, generator):
                self._generator = generator

            def __iter__(self):
                self._generator, generator_copy = tee(self._generator)
                return generator_copy

        trajectories = MockedIterableFromGenerator(trajectories)
        algorithm = SQIL(demonstrations=trajectories, **parameters)
        if "rl_algo" in self.loaded_parameters:
            algorithm.rl_algo = self.loaded_parameters.get("rl_algo")
            algorithm.rl_algo.set_env(vectorized_environment)
            algorithm.rl_algo.replay_buffer.set_demonstrations(trajectories)
        return algorithm

    def train(self, algorithm: SQIL, total_timesteps: int, callback_list: CallbackList):
        algorithm.train(total_timesteps=total_timesteps, callback=callback_list)

    def save_algorithm(self, algorithm: SQIL, folder_path: Path):
        algorithm.rl_algo.save(folder_path / FILE_NAME_SB3_ALGORITHM, exclude=["replay_buffer_kwargs"])

    def load_algorithm(self, folder_path: Path):
        # FIXME: Only works because rl_algo_class is hard-coded to DQN above
        rl_algo = DQN.load(
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

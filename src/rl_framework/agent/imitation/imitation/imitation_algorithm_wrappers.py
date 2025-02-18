import copy
import logging
import math
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Mapping, Optional

import gymnasium
import numpy as np
import torch
from imitation.algorithms.adversarial.airl import AIRL
from imitation.algorithms.adversarial.gail import GAIL
from imitation.algorithms.base import DemonstrationAlgorithm
from imitation.algorithms.bc import BC, BCTrainingMetrics, RolloutStatsComputer
from imitation.algorithms.density import DensityAlgorithm
from imitation.algorithms.sqil import SQIL, SQILReplayBuffer
from imitation.data import buffer
from imitation.data.types import TrajectoryWithRew, TransitionMapping, Transitions
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util import util
from imitation.util.networks import RunningNorm
from stable_baselines3 import DQN, PPO, SAC
from stable_baselines3.common.base_class import BasePolicy
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.dqn.policies import DQNPolicy
from stable_baselines3.sac.policies import SACPolicy

from rl_framework.util import (
    FeaturesExtractor,
    LoggingCallback,
    SavingCallback,
    SizedGenerator,
    add_callbacks_to_callback,
    create_memory_efficient_transition_batcher,
    get_sb3_policy_kwargs_for_features_extractor,
)

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
                        obs_tensor = util.safe_to_tensor(validation_transitions["obs"], device=algorithm.policy.device)
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


class GAILAlgorithmWrapper(AlgorithmWrapper):
    def __init__(self, algorithm_parameters):
        super().__init__(algorithm_parameters)
        self.venv = None

        def patched_set_demonstrations(self, demonstrations: SizedGenerator[TrajectoryWithRew]) -> None:
            self._demo_data_loader = create_memory_efficient_transition_batcher(
                demonstrations,
                self.demo_batch_size,
            )
            self._endless_expert_iterator = self._demo_data_loader

        GAIL.set_demonstrations = patched_set_demonstrations

    def build_algorithm(
        self,
        trajectories: SizedGenerator[TrajectoryWithRew],
        vectorized_environment: VecEnv,
        features_extractor: Optional[FeaturesExtractor] = None,
    ) -> GAIL:
        """
        Build the GAIL algorithm with the given parameters.

        Args:
            trajectories: Trajectories to train the imitation algorithm on.
            vectorized_environment: Vectorized environment (used to construct the reward function by predicting
                 similarity of policy rollouts and expert demonstrations with a continuously updated discriminator)
            features_extractor: Features extractor (preprocessing of observations to vectors, trainable).

        Returns:
            GAIL: GAIL algorithm object initialized with the given parameters.

        """
        self.venv = vectorized_environment
        if features_extractor:
            self.policy_kwargs.update(get_sb3_policy_kwargs_for_features_extractor(features_extractor))
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
        parameters.update(**self.algorithm_parameters)
        algorithm = GAIL(demonstrations=trajectories, **parameters)
        return algorithm

    def train(self, algorithm: GAIL, total_timesteps: int, callback_list: CallbackList, *args, **kwargs):
        add_callbacks_to_callback(callback_list, algorithm.gen_callback)
        algorithm.gen_train_timesteps = min(algorithm.gen_train_timesteps, total_timesteps)
        algorithm._gen_replay_buffer = buffer.ReplayBuffer(
            algorithm.gen_train_timesteps,
            self.venv,
        )
        algorithm.train(total_timesteps=total_timesteps)

    def save_algorithm(self, algorithm: GAIL, folder_path: Path):
        algorithm.gen_algo.save(folder_path / FILE_NAME_SB3_ALGORITHM)
        torch.save(algorithm._reward_net, folder_path / FILE_NAME_REWARD_NET)

    def load_algorithm(self, folder_path: Path):
        gen_algo = self.rl_algo_class.load(folder_path / FILE_NAME_SB3_ALGORITHM)
        reward_net = torch.load(folder_path / FILE_NAME_REWARD_NET)
        self.loaded_parameters.update({"gen_algo": gen_algo, "reward_net": reward_net})


class AIRLAlgorithmWrapper(AlgorithmWrapper):
    def __init__(self, algorithm_parameters):
        super().__init__(algorithm_parameters)
        self.venv = None

        def patched_set_demonstrations(self, demonstrations: SizedGenerator[TrajectoryWithRew]) -> None:
            self._demo_data_loader = create_memory_efficient_transition_batcher(
                demonstrations,
                self.demo_batch_size,
            )
            self._endless_expert_iterator = self._demo_data_loader

        AIRL.set_demonstrations = patched_set_demonstrations

    def build_algorithm(
        self,
        trajectories: SizedGenerator[TrajectoryWithRew],
        vectorized_environment: VecEnv,
        features_extractor: Optional[FeaturesExtractor] = None,
    ) -> AIRL:
        """
        Build the AIRL algorithm with the given parameters.

        Args:
            trajectories: Trajectories to train the imitation algorithm on.
            vectorized_environment: Vectorized environment (used to construct the reward function by predicting
                 similarity of policy rollouts and expert demonstrations with a continuously updated discriminator)
            features_extractor: Features extractor (preprocessing of observations to vectors, trainable).

        Returns:
            AIRL: AIRL algorithm object initialized with the given parameters.

        """
        self.venv = vectorized_environment
        if features_extractor:
            self.policy_kwargs.update(get_sb3_policy_kwargs_for_features_extractor(features_extractor))
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
        parameters.update(**self.algorithm_parameters)
        algorithm = AIRL(demonstrations=trajectories, **parameters)
        return algorithm

    def train(self, algorithm: AIRL, total_timesteps: int, callback_list: CallbackList, *args, **kwargs):
        add_callbacks_to_callback(callback_list, algorithm.gen_callback)
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
        gen_algo = self.rl_algo_class.load(folder_path / FILE_NAME_SB3_ALGORITHM)
        reward_net = torch.load(folder_path / FILE_NAME_REWARD_NET)
        self.loaded_parameters.update({"gen_algo": gen_algo, "reward_net": reward_net})


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


class SQILAlgorithmWrapper(AlgorithmWrapper):
    def __init__(self, algorithm_parameters):
        super().__init__(algorithm_parameters)

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
        features_extractor: Optional[FeaturesExtractor] = None,
    ) -> SQIL:
        """
        Build the SQIL algorithm with the given parameters.

        Args:
            trajectories: Trajectories to train the imitation algorithm on.
            vectorized_environment: Vectorized environment (used to train the RL algorithm, but half of the RL algorithm
                memory keeps filled with expert demonstrations; also all rewards from the live environment are set to
                0.0, while all rewards from the expert demonstrations are set to 1.0).
            features_extractor: Features extractor (preprocessing of observations to vectors, trainable).

        Returns:
            SQIL: SQIL algorithm object initialized with the given parameters.

        """
        # FIXME: SQILReplayBuffer inherits from sb3.ReplayBuffer which doesn't support dict observations.
        #  Maybe it can be patched to inherit from sb3.DictReplayBuffer.
        assert not isinstance(
            vectorized_environment.observation_space, gymnasium.spaces.Dict
        ), "SQILReplayBuffer does not support Dict observation spaces."

        if features_extractor:
            self.policy_kwargs.update(get_sb3_policy_kwargs_for_features_extractor(features_extractor))

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

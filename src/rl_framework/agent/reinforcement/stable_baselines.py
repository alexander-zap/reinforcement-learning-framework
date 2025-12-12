import tempfile
from collections import defaultdict
from copy import deepcopy
from functools import partial
from os import cpu_count
from pathlib import Path
from typing import Dict, List, Optional, Type, Union

import gymnasium
import numpy as np
import pettingzoo
import stable_baselines3
import torch.onnx
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.env_util import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecEnv, VecMonitor
from supersuit.vector import MakeCPUAsyncConstructor, MarkovVectorEnv
from supersuit.vector.sb3_vector_wrapper import SB3VecEnvWrapper

from rl_framework.agent.reinforcement_learning_agent import RLAgent
from rl_framework.util import (
    Connector,
    DummyConnector,
    FeaturesExtractor,
    LoggingCallback,
    SavingCallback,
    get_sb3_policy_kwargs_for_features_extractor,
    wrap_environment_with_features_extractor_preprocessor,
)


class StableBaselinesAgent(RLAgent):
    @property
    def algorithm(self) -> BaseAlgorithm:
        return self._algorithm

    @algorithm.setter
    def algorithm(self, value: BaseAlgorithm):
        self._algorithm = value

    def __init__(
        self,
        algorithm_class: Type[BaseAlgorithm] = stable_baselines3.PPO,
        algorithm_parameters: Optional[Dict] = None,
        features_extractor: Optional[FeaturesExtractor] = None,
    ):
        """
        Initialize an agent which will trained on one of Stable-Baselines3 algorithms.

        Args:
            algorithm_class (Type[BaseAlgorithm]): SB3 RL algorithm class. Specifies the algorithm for RL training.
                Defaults to PPO.
            algorithm_parameters (Dict): Parameters / keyword arguments for the specified SB3 RL Algorithm class.
                See https://stable-baselines3.readthedocs.io/en/master/modules/base.html for details on common params.
                See individual docs (e.g., https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
                for algorithm-specific params.
            features_extractor: When provided, specifies the observation processor to be
                    used before the action/value prediction network.
        """
        super().__init__(algorithm_class, algorithm_parameters, features_extractor)

        self.algorithm_parameters = self._add_required_default_parameters(self.algorithm_parameters)

        additional_parameters = (
            {"_init_setup_model": False} if (getattr(self.algorithm_class, "_setup_model", None)) else {}
        )

        self.algorithm: BaseAlgorithm = self.algorithm_class(
            env=None, **self.algorithm_parameters, **additional_parameters
        )
        self.algorithm_needs_initialization = True

    def train(
        self,
        total_timesteps: int = 100000,
        connector: Optional[Connector] = None,
        training_environments: List[Union[gymnasium.Env, pettingzoo.ParallelEnv, VecEnv]] = None,
        *args,
        **kwargs,
    ):
        """
        Train the instantiated agent on the environment.

        This training is done by using the agent-on-environment training method provided by Stable-baselines3.

        The model is changed in place, therefore the updated model can be accessed in the `.model` attribute
        after the agent has been trained.

        Args:
            training_environments (List[gymnasium.Env, pettingzoo.ParallelEnv]):
                List of environments on which the agent should be trained on.
            total_timesteps (int): Amount of individual steps the agent should take before terminating the training.
            connector (Connector): Connector for executing callbacks (e.g., logging metrics and saving checkpoints)
                on training time. Calls need to be declared manually in the code.
        """

        def make_env(env_list: list, index: int):
            return env_list[index]

        if not training_environments:
            raise ValueError("No training environments have been provided to the train-method.")

        if not connector:
            connector = DummyConnector()

        if self.features_extractor:
            training_environments = [
                wrap_environment_with_features_extractor_preprocessor(env, self.features_extractor)
                for env in training_environments
            ]

        if isinstance(training_environments[0], pettingzoo.ParallelEnv):
            vector_envs = []
            for pettingzoo_environment in training_environments:
                vector_env = MarkovVectorEnv(pettingzoo_environment, black_death=True)
                vector_envs.append(vector_env)

            environment_return_functions = [
                partial(make_env, vector_envs, env_index) for env_index in range(len(vector_envs))
            ]

            vectorized_environment = MakeCPUAsyncConstructor(min(cpu_count(), len(environment_return_functions)))(
                environment_return_functions, vector_envs[0].observation_space, vector_envs[0].action_space
            )

            class AutoResetSB3VecEnvWrapper(SB3VecEnvWrapper):
                """
                A SB3VecEnvWrapper for Pettingzoo based vectorized environments (through MarkovVectorEnv).
                "Automatically resets" episodes and sets infos on step:
                    - `infos["terminal_observation"] = observation` when done
                    - `infos["TimeLimit.truncated"] = True` when truncated (else False)
                """

                def __init__(self, vectorized_environment):
                    super().__init__(vectorized_environment)
                    self.last_observations = None
                    self.last_terminations = None
                    self.last_truncations = None
                    self.last_rewards = None
                    self.last_infos = None
                    self.agents_to_reset = []

                def step_wait(self):
                    observations, rewards, terminations, truncations, infos = self.venv.step_wait()
                    dones = np.array([terminations[i] or truncations[i] for i in range(len(terminations))])

                    observations_to_return = deepcopy(observations)
                    rewards_to_return = deepcopy(rewards)
                    dones_to_return = deepcopy(dones)
                    infos_to_return = deepcopy(infos)

                    for i in range(len(dones)):
                        if i in self.agents_to_reset:
                            rewards_to_return[i] = self.last_rewards[i]
                            dones_to_return[i] = 1
                            infos_to_return[i] = self.last_infos[i]
                            infos_to_return[i]["TimeLimit.truncated"] = (
                                True if self.last_truncations[i] and not self.last_terminations[i] else False
                            )
                            infos_to_return[i]["terminal_observation"] = self.last_observations[i]
                            self.agents_to_reset.remove(i)
                        elif dones[i]:
                            # repeat old observation (workaround; cannot reset agent independently in MarkovVectorEnv)
                            observations_to_return[i] = self.last_observations[i]
                            dones_to_return[i] = self.last_truncations[i] or self.last_terminations[i]
                            rewards_to_return[i] = self.last_rewards[i]
                            infos_to_return[i] = self.last_infos[i]
                            self.agents_to_reset.append(i)

                    self.last_observations = observations
                    self.last_terminations = terminations
                    self.last_truncations = truncations
                    self.last_rewards = rewards
                    self.last_infos = infos

                    return observations_to_return, rewards_to_return, dones_to_return, infos_to_return

            vectorized_environment = AutoResetSB3VecEnvWrapper(vectorized_environment)
            vectorized_environment = VecMonitor(vectorized_environment)

        elif isinstance(training_environments[0], gymnasium.Env):
            training_environments = [Monitor(env) for env in training_environments]
            environment_return_functions = [
                partial(make_env, training_environments, env_index) for env_index in range(len(training_environments))
            ]

            # noinspection PyCallingNonCallable
            vectorized_environment = self.to_vectorized_env(env_fns=environment_return_functions)

        elif isinstance(training_environments[0], VecEnv):
            assert len(training_environments) == 1
            vectorized_environment = training_environments[0]

        else:
            raise TypeError(f"Environment type {type(training_environments[0])} not supported!")

        algorithm_kwargs = {"env": vectorized_environment}
        if self.algorithm_needs_initialization:
            parameters = defaultdict(dict, {**self.algorithm_parameters})
            if self.features_extractor:
                parameters["policy_kwargs"].update(
                    get_sb3_policy_kwargs_for_features_extractor(self.features_extractor)
                )
            algorithm_kwargs.update(parameters)
            self.algorithm = self.algorithm_class(**algorithm_kwargs)
            self.algorithm_needs_initialization = False
        else:
            with tempfile.TemporaryDirectory("w") as tmp_dir:
                tmp_path = Path(tmp_dir) / "tmp_model.zip"
                self.save_to_file(tmp_path)
                algorithm_kwargs["path"] = tmp_path
                algorithm_kwargs["custom_objects"] = self.algorithm_parameters
                # noinspection PyUnresolvedReferences
                self.algorithm = self.algorithm_class.load(**algorithm_kwargs)

        callback_list = CallbackList([SavingCallback(self, connector), LoggingCallback(connector)])
        self.algorithm.learn(total_timesteps=total_timesteps, callback=callback_list)
        vectorized_environment.close()

    def to_vectorized_env(self, env_fns) -> VecEnv:
        return SubprocVecEnv(env_fns)

    def choose_action(self, observation: object, deterministic: bool = False, *args, **kwargs):
        """
        Chooses action which the agent will perform next, according to the observed environment.

        Args:
            observation (object): Observation of the environment
            deterministic (bool): Whether the action should be determined in a deterministic or stochastic way.

        Returns: action: Action to take according to policy.

        """

        (
            action,
            _,
        ) = self.algorithm.predict(
            observation,
            deterministic=deterministic,
        )
        if not action.shape:
            action = action.item()
        return action

    def save_as_onnx(self, file_path: Path) -> None:
        """Save the agent as ONNX model.

        Args:
            file_path (Path): The file where the agent should be saved to.
        """
        assert str(file_path).endswith(".onnx"), "File path must end with .onnx"

        observation_size = self.algorithm.observation_space.shape
        dummy_input = torch.randn(1, *observation_size)
        torch.onnx.export(self.algorithm.policy, dummy_input, file_path, opset_version=17, input_names=["input"])

    def save_to_file(self, file_path: Path, *args, **kwargs) -> None:
        """Save the agent to a file (for later loading).

        Args:
            file_path (Path): The file where the agent should be saved to (SB3 expects a file name ending with .zip).
        """
        self.algorithm.save(file_path)

    def load_from_file(self, file_path: Path, algorithm_parameters: Dict = None, *args, **kwargs) -> None:
        """Load the agent in-place from an agent-save folder.

        Args:
            file_path (Path): The model filename (file ending with .zip).
            algorithm_parameters: Parameters to be set for the loaded algorithm.
                Providing None leads to keeping the previously set parameters.
        """
        if algorithm_parameters:
            self.algorithm_parameters = self._add_required_default_parameters(algorithm_parameters)
        self.algorithm = self.algorithm_class.load(path=file_path, env=None, **self.algorithm_parameters)
        self.algorithm_needs_initialization = False

    @staticmethod
    def _add_required_default_parameters(algorithm_parameters: Optional[Dict]):
        """
        Add missing required parameters to `algorithm_parameters`.
        Required parameters currently are:
            - "policy": needs to be set for every BaseRLAlgorithm. Set to "MlpPolicy" if not provided.
            - "tensorboard_log": needs to be set for logging callbacks. Set to newly created temp dir if not provided.

        Args:
            algorithm_parameters (Optional[Dict]): Parameters passed by user (in .__init__ or .load_from_file).

        Returns:
            algorithm_parameters (Dict): Parameter dictionary with filled up default parameter entries

        """
        if "policy" not in algorithm_parameters:
            algorithm_parameters.update({"policy": "MlpPolicy"})

        # Existing tensorboard log paths can be used (e.g., for continuing training of downloaded agents).
        # If not provided, tensorboard will be logged to newly created temp dir.
        if "tensorboard_log" not in algorithm_parameters:
            tensorboard_log_path = tempfile.mkdtemp()
            algorithm_parameters.update({"tensorboard_log": tensorboard_log_path})

        return algorithm_parameters

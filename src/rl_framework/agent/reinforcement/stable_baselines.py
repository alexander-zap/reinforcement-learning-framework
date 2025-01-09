import tempfile
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Type

import gymnasium
import numpy as np
import stable_baselines3
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.env_util import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from rl_framework.agent.reinforcement.reinforcement_learning_agent import RLAgent
from rl_framework.util import Connector, LoggingCallback, SavingCallback


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
        algorithm_parameters: Dict = None,
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
        """
        self.algorithm_class: Type[BaseAlgorithm] = algorithm_class

        self.algorithm_parameters = self._add_required_default_parameters(algorithm_parameters)

        additional_parameters = (
            {"_init_setup_model": False} if (getattr(self.algorithm_class, "_setup_model", None)) else {}
        )

        self.algorithm: BaseAlgorithm = self.algorithm_class(
            env=None, **self.algorithm_parameters, **additional_parameters
        )
        self.algorithm_needs_initialization = True

    def train(
        self,
        total_timesteps: int,
        connector: Connector,
        training_environments: List[gymnasium.Env] = None,
        *args,
        **kwargs,
    ):
        """
        Train the instantiated agent on the environment.

        This training is done by using the agent-on-environment training method provided by Stable-baselines3.

        The model is changed in place, therefore the updated model can be accessed in the `.model` attribute
        after the agent has been trained.

        Args:
            training_environments (List[gymnasium.Env]): List of environments on which the agent should be trained on.
            total_timesteps (int): Amount of individual steps the agent should take before terminating the training.
            connector (Connector): Connector for executing callbacks (e.g., logging metrics and saving checkpoints)
                on training time. Calls need to be declared manually in the code.
        """

        def make_env(index: int):
            return training_environments[index]

        if not training_environments:
            raise ValueError("No training environments have been provided to the train-method.")

        training_environments = [Monitor(env) for env in training_environments]
        environment_return_functions = [partial(make_env, env_index) for env_index in range(len(training_environments))]

        # noinspection PyCallingNonCallable
        vectorized_environment = self.to_vectorized_env(env_fns=environment_return_functions)

        if self.algorithm_needs_initialization:
            self.algorithm = self.algorithm_class(env=vectorized_environment, **self.algorithm_parameters)
            self.algorithm_needs_initialization = False
        else:
            with tempfile.TemporaryDirectory("w") as tmp_dir:
                tmp_path = Path(tmp_dir) / "tmp_model.zip"
                self.save_to_file(tmp_path)
                self.algorithm = self.algorithm_class.load(
                    path=tmp_path, env=vectorized_environment, custom_objects=self.algorithm_parameters
                )

        callback_list = CallbackList(
            [SavingCallback(self, connector, training_environments[0]), LoggingCallback(connector)]
        )
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

        Returns: action (int): Action to take according to policy.

        """

        # SB3 model expects multiple observations as input and will output an array of actions as output
        (
            action,
            _,
        ) = self.algorithm.predict(
            np.array([observation]),
            deterministic=deterministic,
        )
        return action[0]

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
        if algorithm_parameters is None:
            algorithm_parameters = {}

        if "policy" not in algorithm_parameters:
            algorithm_parameters.update({"policy": "MlpPolicy"})

        # Existing tensorboard log paths can be used (e.g., for continuing training of downloaded agents).
        # If not provided, tensorboard will be logged to newly created temp dir.
        if "tensorboard_log" not in algorithm_parameters:
            tensorboard_log_path = tempfile.mkdtemp()
            algorithm_parameters.update({"tensorboard_log": tensorboard_log_path})

        return algorithm_parameters

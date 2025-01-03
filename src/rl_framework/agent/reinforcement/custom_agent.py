from pathlib import Path
from typing import Dict, List, Optional, Type

import gymnasium as gym

from rl_framework.agent.reinforcement.custom_algorithms import (
    CustomAlgorithm,
    QLearning,
)
from rl_framework.agent.reinforcement_learning_agent import RLAgent
from rl_framework.util import Connector


class CustomAgent(RLAgent):
    @property
    def algorithm(self) -> CustomAlgorithm:
        return self._algorithm

    @algorithm.setter
    def algorithm(self, value: CustomAlgorithm):
        self._algorithm = value

    def __init__(
        self,
        algorithm_class: Type[CustomAlgorithm] = QLearning,
        algorithm_parameters: Optional[Dict] = None,
    ):
        """
        Initialize an agent which will trained on one of custom implemented algorithms.

        Args:
            algorithm_class (Type[CustomAlgorithm]): Class of custom implemented Algorithm.
                Specifies the algorithm for RL training.
                Defaults to Q-Learning.
            algorithm_parameters (Dict): Parameters / keyword arguments for the specified Algorithm class.
        """

        if algorithm_parameters is None:
            algorithm_parameters = {}

        self.algorithm = algorithm_class(**algorithm_parameters)

    def train(
        self,
        total_timesteps: int,
        connector: Connector,
        training_environments: List[gym.Env] = None,
        *args,
        **kwargs,
    ):
        """
        Train the instantiated agent on the environment.

        This training is done by using the agent-on-environment training method provided by the custom algorithm.

        Args:
            training_environments (List[gym.Env]): Environment on which the agent should be trained on.
                If n_environments is set above 1, multiple environments enables parallel training of an agent.
            total_timesteps (int): Amount of individual steps the agent should take before terminating the training.
            connector (Connector): Connector for executing callbacks (e.g., logging metrics and saving checkpoints)
                on training time. Logging is executed by calling the connector.log method.
                Calls need to be declared manually in the code.
        """

        if not training_environments:
            raise ValueError("No training environments have been provided to the train-method.")

        self.algorithm.train(
            connector=connector,
            training_environments=training_environments,
            total_timesteps=total_timesteps,
            *args,
            **kwargs,
        )

    def choose_action(self, observation: object, deterministic: bool = False, *args, **kwargs):
        """
        Chooses action which the agent will perform next, according to the observed environment.

        Args:
            observation (object): Observation of the environment
            deterministic (bool): Whether the action should be determined in a deterministic or stochastic way.

        Returns: action (int): Action to take according to policy.

        """

        return self.algorithm.choose_action(observation=observation, deterministic=deterministic, *args, **kwargs)

    def save_to_file(self, file_path: Path, *args, **kwargs):
        """Save the agent to a file at a certain path (to be loadable again later).

        Args:
            file_path: File where the agent should be saved to.
        """
        self.algorithm.save_to_file(file_path=file_path)

    def load_from_file(self, file_path: Path, algorithm_parameters: Optional[Dict] = None, *args, **kwargs):
        """Load the agent from a previously save agent-file.

        Args:
            file_path: File where the agent-file to be loaded is located at.
            algorithm_parameters (Optional[Dict]): Parameters which should overwrite the algorithm after loading.
        """
        self.algorithm.load_from_file(file_path, algorithm_parameters=algorithm_parameters)

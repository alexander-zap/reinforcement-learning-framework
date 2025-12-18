from pathlib import Path
from typing import Dict, List, Optional, Type

from rl_framework.agent.reinforcement.custom_algorithms import (
    CustomAlgorithm,
    QLearning,
)
from rl_framework.agent.reinforcement_learning_agent import RLAgent
from rl_framework.util import (
    Connector,
    DummyConnector,
    Environment,
    FeaturesExtractor,
    wrap_environment_with_features_extractor_preprocessor,
)


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
        features_extractor: Optional[FeaturesExtractor] = None,
    ):
        """
        Initialize an agent which will trained on one of custom implemented algorithms.

        Args:
            algorithm_class (Type[CustomAlgorithm]): Class of custom implemented Algorithm.
                Specifies the algorithm for RL training.
                Defaults to Q-Learning.
            algorithm_parameters (Dict): Parameters / keyword arguments for the specified Algorithm class.
            features_extractor: When provided, specifies the observation processor to be
                    used before the action/value prediction network.
        """

        super().__init__(algorithm_class, algorithm_parameters, features_extractor)

        self.algorithm = self.algorithm_class(**self.algorithm_parameters, features_extractor=self.features_extractor)

    def train(
        self,
        total_timesteps: int,
        connector: Optional[Connector] = None,
        training_environments: List[Environment] = None,
        *args,
        **kwargs,
    ):
        """
        Train the instantiated agent on the environment.

        This training is done by using the agent-on-environment training method provided by the custom algorithm.

        Args:
            training_environments (List[Environment]): Environment on which the agent should be
                trained on. Multiple environments enables parallel training of an agent.
            total_timesteps (int): Amount of individual steps the agent should take before terminating the training.
            connector (Connector): Connector for executing callbacks (e.g., logging metrics and saving checkpoints)
                on training time. Logging is executed by calling the connector.log method.
                Calls need to be declared manually in the code.
        """

        if not training_environments:
            raise ValueError("No training environments have been provided to the train-method.")

        if not connector:
            connector = DummyConnector()

        if self.features_extractor:
            training_environments = [
                wrap_environment_with_features_extractor_preprocessor(env, self.features_extractor)
                for env in training_environments
            ]

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

    def save_as_onnx(self, file_path: Path, *args, **kwargs) -> None:
        raise NotImplementedError

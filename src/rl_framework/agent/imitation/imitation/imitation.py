import shutil
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Type

import gymnasium
from imitation.algorithms.adversarial.airl import AIRL
from imitation.algorithms.adversarial.gail import GAIL
from imitation.algorithms.base import DemonstrationAlgorithm
from imitation.algorithms.bc import BC
from imitation.algorithms.density import DensityAlgorithm
from imitation.algorithms.sqil import SQIL
from stable_baselines3.common.env_util import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from rl_framework.agent.imitation.episode_sequence import EpisodeSequence
from rl_framework.agent.imitation.imitation.imitation_algorithm_wrappers import (
    AIRLAlgorithmWrapper,
    AlgorithmWrapper,
    BCAlgorithmWrapper,
    DensityAlgorithmWrapper,
    GAILAlgorithmWrapper,
    SQILAlgorithmWrapper,
)
from rl_framework.agent.imitation.imitation_learning_agent import ILAgent
from rl_framework.util import Connector

IMITATION_ALGORITHM_WRAPPER_REGISTRY = {
    BC: BCAlgorithmWrapper,
    GAIL: GAILAlgorithmWrapper,
    AIRL: AIRLAlgorithmWrapper,
    DensityAlgorithm: DensityAlgorithmWrapper,
    SQIL: SQILAlgorithmWrapper,
}


class ImitationAgent(ILAgent):
    @property
    def algorithm(self) -> DemonstrationAlgorithm:
        return self._algorithm

    @algorithm.setter
    def algorithm(self, value: DemonstrationAlgorithm):
        self._algorithm = value

    def __init__(
        self,
        algorithm_class: Type[DemonstrationAlgorithm] = BC,
        algorithm_parameters: Dict = None,
    ):
        """
        Initialize an agent which will trained on one of imitation algorithms.

        Args:
            algorithm_class (Type[BaseAlgorithm]): SB3 RL algorithm class. Specifies the algorithm for RL training.
                Defaults to PPO.
            algorithm_parameters (Dict): Parameters / keyword arguments for the specified imitation algorithm class.
                See https://imitation.readthedocs.io/en/latest/_api/imitation.algorithms.base.html for details on
                    common params.
                See individual docs (e.g., https://imitation.readthedocs.io/en/latest/algorithms/bc.html)
                for algorithm-specific params.
        """
        self.algorithm_wrapper: AlgorithmWrapper = IMITATION_ALGORITHM_WRAPPER_REGISTRY[algorithm_class]()
        self.algorithm_parameters = self._add_required_default_parameters(algorithm_parameters)
        self.algorithm = None
        self.algorithm_policy = None

    def train(
        self,
        total_timesteps: int,
        connector: Connector,
        episode_sequence: EpisodeSequence = None,
        training_environments: List[gymnasium.Env] = None,
        *args,
        **kwargs,
    ):
        """
        Train the instantiated agent on a list of trajectories.

        This training is done by using imitation learning policies, provided by the imitation library.

        The model is changed in place, therefore the updated model can be accessed in the `.model` attribute
        after the agent has been trained.

        Args:
            total_timesteps (int): Amount of (recorded) timesteps to train the agent on.
            episode_sequence (EpisodeSequence): List of episodes on which the agent should be trained on.
            training_environments (List): List of environments
                Required for interaction or attribute extraction (e.g., action/observation space) for some algorithms
            connector (Connector): Connector for executing callbacks (e.g., logging metrics and saving checkpoints)
                on training time. Calls need to be declared manually in the code.
        """

        def make_env(index: int):
            return training_environments[index]

        if not episode_sequence:
            raise ValueError("No transitions have been provided to the train-method.")

        trajectories = episode_sequence.to_imitation_episodes()

        assert len(training_environments) > 0, (
            "All imitation algorithms require an environment to be passed. "
            "Some for parameter definition (e.g., BC), some for active interaction (e.g., SQIL)."
        )
        training_environments = [Monitor(env) for env in training_environments]
        environment_return_functions = [partial(make_env, env_index) for env_index in range(len(training_environments))]
        vectorized_environment = self.to_vectorized_env(env_fns=environment_return_functions)

        if not self.algorithm:
            self.algorithm = self.algorithm_wrapper.build_algorithm(
                self.algorithm_parameters, total_timesteps, trajectories, vectorized_environment
            )
        else:
            self.algorithm.set_demonstrations(trajectories)

        self.algorithm_wrapper.train(self.algorithm, total_timesteps)

        self.algorithm_policy = self.algorithm.policy

        vectorized_environment.close()

    @staticmethod
    def to_vectorized_env(env_fns) -> VecEnv:
        return SubprocVecEnv(env_fns)

    def choose_action(self, observation: object, deterministic: bool = False, *args, **kwargs):
        """
        Chooses action which the agent will perform next, according to the observed environment.

        Args:
            observation (object): Observation of the environment
            deterministic (bool): Whether the action should be determined in a deterministic or stochastic way.

        Returns: action (int): Action to take according to policy.

        """

        if not self.algorithm_policy:
            raise ValueError("Cannot predict action for uninitialized agent. Start a training first to initialize.")

        # SB3 model expects multiple observations as input and will output an array of actions as output
        (
            action,
            _,
        ) = self.algorithm_policy.predict(
            [observation],
            deterministic=deterministic,
        )
        return action[0]

    def save_to_file(self, file_path: Path, *args, **kwargs) -> None:
        """Save the agent to a file (for later loading).

        Args:
            file_path (Path): The path where the agent should be saved to (expects a .zip directory).
        """

        assert str(file_path).endswith(".zip")
        folder_path = Path(str(file_path)[:-4])
        folder_path.mkdir(parents=True, exist_ok=True)

        if not self.algorithm and not self.algorithm_policy:
            raise AttributeError(
                "Trying to save non-initialized imitation algorithm and non-initialized policy. "
                "Call the train method first."
            )
        elif not self.algorithm:
            raise AttributeError(
                "Trying to save non-initialized imitation algorithm. "
                "This likely is caused by trying to save a loaded model without re-training."
            )
        else:
            self.algorithm_wrapper.save_to_file(self.algorithm, folder_path)

        shutil.make_archive(str(folder_path), "zip", str(folder_path))

    def load_from_file(self, file_path: Path, algorithm_parameters: Dict = None, *args, **kwargs) -> None:
        """Loads the agent policy (`self.agent_policy`) in-place from a zipped folder.
        The agent algorithm (`self.algorithm`) is re-initialized in the train method and remains None until then.

        Args:
            file_path (Path): The file path the agent has been saved to before.
            algorithm_parameters: Parameters to be set for the loaded algorithm.
                Providing None leads to keeping the previously set parameters.
        """
        assert str(file_path).endswith(".zip")

        self.algorithm = None

        if algorithm_parameters:
            self.algorithm_parameters.update(**algorithm_parameters)

        folder_path = Path(str(file_path)[:-4])
        shutil.unpack_archive(file_path, folder_path, "zip")
        self.algorithm_policy = self.algorithm_wrapper.load_from_file(folder_path)

    @staticmethod
    def _add_required_default_parameters(algorithm_parameters: Optional[Dict]):
        """
        Add missing required parameters to `algorithm_parameters`.
        Required parameters currently are:
            - "allow_variable_horizon": Allow using gym environments with variable episode lengths.
                The `imitation` library discourages this, because algorithms are able to exploit this information.
                See https://imitation.readthedocs.io/en/latest/main-concepts/variable_horizon.html
                Nevertheless, we do not want the `imitation` library to not interfere with raising errors.
                The user needs to explicitly enable this.

        Args:
            algorithm_parameters (Optional[Dict]): Parameters passed by user (in .__init__ or .load_from_file).

        Returns:
            algorithm_parameters (Dict): Parameter dictionary with filled up default parameter entries

        """
        if algorithm_parameters is None:
            algorithm_parameters = {}

        if "allow_variable_horizon" not in algorithm_parameters:
            algorithm_parameters["allow_variable_horizon"] = True

        return algorithm_parameters
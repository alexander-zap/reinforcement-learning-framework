import logging
import os
import pickle
import random
from pathlib import Path
from typing import Dict, List, Optional, Union

import gymnasium as gym
import numpy as np
import pettingzoo
from tqdm import tqdm

from rl_framework.agent.reinforcement.custom_algorithms.base_custom_algorithm import (
    CustomAlgorithm,
)
from rl_framework.util import (
    Connector,
    FeaturesExtractor,
    encode_observations_with_features_extractor,
)


class QLearning(CustomAlgorithm):
    @property
    def q_table(self):
        return self._q_table

    @q_table.setter
    def q_table(self, value):
        self._q_table = value

    def __init__(
        self,
        n_actions: int,
        n_observations: int,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        randomize_q_table: bool = True,
        features_extractor: Optional[FeaturesExtractor] = None,
    ):
        """
        Initialize an Q-Learning agent which will be trained.
        """
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.n_actions = n_actions
        self.features_extractor = features_extractor

        if randomize_q_table:
            self.q_table = np.random.random_sample((n_observations, n_actions)) * 0.1
        else:
            self.q_table = np.full((n_observations, n_actions), 0.0)

    def _update_q_table(
        self,
        prev_observation: object,
        prev_action: int,
        observation: object,
        reward: float,
    ):
        """
        Update _q_table based on previous observation, previous action, new observation and received reward

        Args:
            prev_observation (object): Previous observation (St)
            prev_action (in): Previous action (at)
            observation (object): New observation (St+1) after executing action at in state St
            reward (float): Reward for executing action at in state St

        """
        if self.features_extractor:
            # [0][0] since we expect a single discretized value for Q-Learning
            prev_observation = encode_observations_with_features_extractor([prev_observation], self.features_extractor)[
                0
            ][0]
            observation = encode_observations_with_features_extractor([observation], self.features_extractor)[0][0]

        q_old = self._q_table[prev_observation, prev_action]
        q_new = (1 - self.alpha) * q_old + self.alpha * (reward + self.gamma * np.max(self._q_table[observation]))
        self._q_table[prev_observation, prev_action] = q_new

    def choose_action(self, observation: object, deterministic: bool = False, *args, **kwargs) -> int:
        """
        Chooses action which the agent will perform next, according to the observed environment.

        Args:
            observation (object): Observation of the environment
            deterministic (bool): Whether the action should be determined in a deterministic or stochastic way.
                NOTE: The Q-Table does not support stochastic action choice.

        Returns: action (int): Action to take according to policy.

        """
        if self.features_extractor:
            # [0][0] since we expect a single discretized value for Q-Learning
            observation = encode_observations_with_features_extractor([observation], self.features_extractor)[0][0]

        return np.argmax(self._q_table[observation])

    def train(
        self,
        connector: Connector,
        training_environments: List[Union[gym.Env, pettingzoo.ParallelEnv]],
        total_timesteps: int,
        *args,
        **kwargs,
    ):
        """
        Train the instantiated agent on the environment.

        This training is done by using the Q-Learning method.

        The Q-table is changed in place, therefore the updated Q-table can be accessed in the `.q_table` attribute
        after the agent has been trained.

        Args:
            training_environments (List[Environment]): List of environments on which the agent should be trained on.
                # NOTE: This class only supports training on one single-agent environment.
            total_timesteps (int): Number of timesteps the agent should train for before terminating the training.
            connector (Connector): Connector for executing callbacks (e.g., logging metrics and saving checkpoints)
                on training time. Logging is executed by calling the connector.log method.
                Calls need to be declared manually in the code.e.
        """

        # TODO: Exploration-exploitation strategy is currently hard-coded as epsilon-greedy.
        #   Instead: Pass exploration-exploitation strategy from outside.
        #   See KerasRL for example:
        #         policy = LinearAnnealedPolicy(
        #             EpsGreedyQPolicy(),
        #             attr="eps",
        #             value_max=1.0,
        #             value_min=0.1,
        #             value_test=0.01,
        #             nb_steps=1000000
        #         )
        #         agent = DQNAgent(
        #             model=model,
        #             policy=policy,
        #             ...
        #         )
        def choose_action_according_to_exploration_exploitation_strategy(obs):
            greedy_action = self.choose_action(obs)
            # Choose random action with probability epsilon
            if random.random() < self.epsilon:
                return random.randrange(self.n_actions)
            # Greedy action is chosen with probability (1 - epsilon)
            else:
                return greedy_action

        if len(training_environments) > 1:
            logging.warning(
                f"Reinforcement Learning algorithm {self.__class__.__qualname__} does not support "
                f"training on multiple environments in parallel. Continuing with one environment as "
                f"training environment."
            )
        elif not isinstance(training_environments[0], gym.Env):
            raise ValueError(
                f"Reinforcement Learning algorithm {self.__class__.__qualname__} currently does not support "
                f"training on environment of type {type(training_environments[0])}."
            )

        training_environment = training_environments[0]
        tqdm_progress_bar = tqdm(total=total_timesteps)
        current_timestep = 0

        while current_timestep <= total_timesteps:
            done = False
            episode_reward = 0
            episode_timestep = 0
            prev_observation, _ = training_environment.reset()
            prev_action = choose_action_according_to_exploration_exploitation_strategy(prev_observation)

            while not done:
                episode_timestep += 1
                (
                    observation,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = training_environment.step(prev_action)
                done = terminated or truncated
                action = choose_action_according_to_exploration_exploitation_strategy(observation)
                episode_reward += reward
                # TODO: Replay sampling strategy is currently hard-coded as on-line.
                #   Instead: Pass replay sampling strategy from outside (as Memory-class).
                self._update_q_table(prev_observation, prev_action, observation, float(reward))

                prev_observation = observation
                prev_action = action

                if done:
                    current_timestep += episode_timestep
                    tqdm_progress_bar.n = current_timestep if current_timestep <= total_timesteps else total_timesteps
                    tqdm_progress_bar.refresh()
                    # Gradually reduce epsilon after every done episode
                    self.epsilon = (
                        1.0 - (2.0 * current_timestep / total_timesteps)
                        if self.epsilon > self.epsilon_min
                        else self.epsilon
                    )

                    if connector:
                        connector.log_value_with_timestep(current_timestep, episode_reward, "Episode reward")
                        connector.log_value_with_timestep(current_timestep, self.epsilon, "Epsilon")

        tqdm_progress_bar.close()

    def save_to_file(self, file_path: Path, *args, **kwargs):
        """
        Save the action-prediction model (Q-Table) of the agent to pickle file.

        Args:
            file_path (Text): Path where the model should be saved to.
        """
        # TODO: Save other parameters as well (do not only save Q-table)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(self.q_table, f)

    def load_from_file(self, file_path: Path, algorithm_parameters: Optional[Dict] = None, *args, **kwargs):
        """
        Load the action-prediction model (Q-Table) from a previously created (by the .save function) pickle file.

         Args:
            file_path (Text): Path where the model has been previously saved to.
            algorithm_parameters (Optional[Dict]): Parameters to be set for the downloaded agent-algorithm.
        """
        with open(file_path, "rb") as f:
            self.q_table = pickle.load(f)

        if algorithm_parameters:
            for key, value in algorithm_parameters.items():
                if hasattr(self, key):
                    setattr(self, key, value)

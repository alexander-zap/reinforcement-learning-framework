from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type, Union

import gymnasium as gym
import numpy as np
import pettingzoo
from tqdm import tqdm

from rl_framework.util import (
    Connector,
    FeaturesExtractor,
    wrap_environment_with_features_extractor_preprocessor,
)


class Agent(ABC):
    @property
    @abstractmethod
    def algorithm(self):
        return NotImplementedError

    def __init__(
        self,
        algorithm_class: Type,
        algorithm_parameters: Optional[Dict],
        features_extractor: Optional[FeaturesExtractor],
        *args,
        **kwargs,
    ):
        """
        Initialize an agent.

        Args:
            algorithm_class: Reinforcement or imitation learning class to be used for training the agent.
            algorithm_parameters: Parameters for the specified algorithm class.
            features_extractor: When provided, specifies the observation processor to be
                    used before the action/value prediction network.
        """
        self.algorithm_class = algorithm_class
        self.algorithm_parameters = algorithm_parameters if algorithm_parameters else {}
        self.features_extractor = features_extractor

    def evaluate(
        self,
        evaluation_environment: Union[gym.Env, pettingzoo.ParallelEnv],
        n_eval_episodes: int,
        seeds: Optional[List[int]] = None,
        deterministic: bool = False,
    ) -> Tuple[float, float]:
        """
        Evaluate the agent for ``n_eval_episodes`` episodes and returns average reward and std of reward.

        Args:
            evaluation_environment (gym.Env or pettingzoo.ParallelEnv): The evaluation environment.
            n_eval_episodes (int): Number of episode to evaluate the agent.
            seeds (Optional[List[int]]): List of seeds for evaluations.
                No seed is used if not provided or fewer seeds are provided then n_eval_episodes.
            deterministic (bool): Whether the agents' actions should be determined in a deterministic or stochastic way.
        """

        if self.features_extractor:
            evaluation_environment = wrap_environment_with_features_extractor_preprocessor(
                evaluation_environment, self.features_extractor
            )

        episode_rewards = []

        if isinstance(evaluation_environment, pettingzoo.ParallelEnv):
            prev_observations, _ = evaluation_environment.reset()
            prev_actions = {
                agent: self.choose_action(prev_observations[agent], deterministic=deterministic)
                for agent in evaluation_environment.agents
            }

            episode_reward = {agent: 0.0 for agent in evaluation_environment.agents}

            with tqdm(total=n_eval_episodes) as pbar:
                while len(episode_rewards) < n_eval_episodes:
                    (
                        observations,
                        rewards,
                        terminations,
                        truncations,
                        infos,
                    ) = evaluation_environment.step(prev_actions)

                    terms = np.fromiter(terminations.values(), dtype=bool)
                    truncs = np.fromiter(truncations.values(), dtype=bool)
                    dones = terms | truncs
                    env_done = dones.all()

                    # next action to be executed (based on new observation)
                    actions = {
                        agent: self.choose_action(observations[agent], deterministic=deterministic)
                        for agent in evaluation_environment.agents
                    }

                    for agent in rewards.keys():
                        if agent not in episode_reward and not (terminations[agent] or truncations[agent]):
                            episode_reward[agent] = rewards[agent]
                        elif agent in episode_reward:
                            episode_reward[agent] += rewards[agent]

                    prev_actions = actions

                    if dones.any():
                        done_indices = np.where(dones == True)[0]
                        for done_index in done_indices:
                            agent = list(terminations.keys())[done_index]
                            if agent in episode_reward:
                                episode_rewards.append(episode_reward[agent])
                                pbar.update(1)
                                del episode_reward[agent]

                    if env_done:
                        prev_observations, _ = evaluation_environment.reset()
                        episode_reward = {agent: 0.0 for agent in evaluation_environment.agents}

        else:
            if seeds is None:
                seeds = []

            for episode in tqdm(range(n_eval_episodes)):
                seed = seeds[episode] if episode < len(seeds) else None
                episode_reward = 0

                prev_observation, _ = evaluation_environment.reset(seed=seed)
                prev_action = self.choose_action(prev_observation, deterministic=deterministic)

                while True:
                    (
                        observation,
                        reward,
                        terminated,
                        truncated,
                        info,
                    ) = evaluation_environment.step(prev_action)
                    done = terminated or truncated
                    # next action to be executed (based on new observation)
                    action = self.choose_action(observation, deterministic=deterministic)
                    episode_reward += reward
                    prev_action = action

                    if done:
                        episode_rewards.append(episode_reward)
                        break

        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        return mean_reward, std_reward

    @abstractmethod
    def choose_action(self, observation: object, deterministic: bool, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def save_as_onnx(self, file_path: Path, *args, **kwargs) -> None:
        """
        Save the agent as an ONNX model.

        Args:
            file_path (Path): Path to save the ONNX model.
        """
        raise NotImplementedError

    @abstractmethod
    def save_to_file(self, file_path: Path, *args, **kwargs) -> None:
        raise NotImplementedError

    @abstractmethod
    def load_from_file(self, file_path: Path, algorithm_parameters: Optional[Dict], *args, **kwargs) -> None:
        raise NotImplementedError

    def upload(self, connector: Connector, video_recording_environment: Optional[gym.Env] = None) -> None:
        """
        Evaluate and upload the decision-making agent to the connector.
            Additional option: Generate a video of the agent interacting with the environment.

        Args:
            connector (Connector): Connector for uploading.
            video_recording_environment (gym.Env): Environment used for clip creation before upload.
                Optional. If not provided, no video will be recorded.
        """
        connector.upload(
            agent=self,
            video_recording_environment=video_recording_environment,
        )

    def download(
        self,
        connector: Connector,
        algorithm_parameters: Optional[Dict] = None,
    ):
        """
        Download a previously saved decision-making agent from the connector and replace the `self` agent instance
            in-place with the newly downloaded saved-agent.

        NOTE: Agent and Algorithm class need to be the same as the saved agent.

        Args:
            connector: Connector for downloading.
            algorithm_parameters (Optional[Dict]): Parameters to be set for the downloaded agent.
        """

        # Get the model from the Hub, download and cache the model on your local disk
        agent_file_path = connector.download()
        self.load_from_file(agent_file_path, algorithm_parameters)

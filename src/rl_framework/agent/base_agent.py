import threading
from abc import ABC, abstractmethod
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type

import gymnasium as gym
import numpy as np
import pettingzoo
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv
from tqdm import tqdm

from rl_framework.util import (
    Connector,
    DummyConnector,
    Environment,
    FeaturesExtractor,
    MetricAggregator,
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
        evaluation_environments: List[Environment],
        n_eval_episodes: int,
        connector: Connector = DummyConnector(),
        deterministic: bool = False,
    ) -> Tuple[float, float]:
        """
        Evaluate the agent for ``n_eval_episodes`` episodes and returns average reward and std of reward.

        Args:
            evaluation_environments (List[Environment]): The evaluation environments.
            n_eval_episodes (int): Number of episode to evaluate the agent.
            connector (Connector): Connector for logging evaluation metrics.
            deterministic (bool): Whether the agents' actions should be determined in a deterministic or stochastic way.
        """

        def envs_to_dummy_vec_env(environments: List[gym.Env]) -> VecEnv:
            def make_env(env_list: list, index: int):
                return env_list[index]

            environment_fns = [partial(make_env, environments, env_index) for env_index in range(len(environments))]

            # noinspection PyCallingNonCallable
            vectorized_environment = DummyVecEnv(env_fns=environment_fns)

            return vectorized_environment

        if self.features_extractor:
            evaluation_environments = [
                wrap_environment_with_features_extractor_preprocessor(evaluation_environment, self.features_extractor)
                for evaluation_environment in evaluation_environments
            ]

        episode_rewards = []

        with tqdm(total=n_eval_episodes) as pbar:
            if isinstance(evaluation_environments[0], pettingzoo.ParallelEnv):

                def evaluate_agent_on_environment(evaluation_environment, n_episodes: int):
                    prev_observations, _ = evaluation_environment.reset()
                    prev_actions = {
                        agent: self.choose_action(prev_observations[agent], deterministic=deterministic)
                        for agent in evaluation_environment.agents
                    }

                    episode_reward = {agent: 0.0 for agent in evaluation_environment.agents}

                    while len(episode_rewards) < n_episodes:
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
                            prev_actions = {
                                agent: self.choose_action(prev_observations[agent], deterministic=deterministic)
                                for agent in evaluation_environment.agents
                            }
                            episode_reward = {agent: 0.0 for agent in evaluation_environment.agents}

            elif (
                isinstance(evaluation_environments[0], gym.Env)
                or isinstance(evaluation_environments[0], VecEnv)
                or isinstance(evaluation_environments[0], tuple)
            ):
                # tuple = EnvironmentFactory in format (stub_environment, env_return_function)
                if isinstance(evaluation_environments[0], tuple):
                    environments_from_callable = []
                    for _, env_func in evaluation_environments:
                        instantiated_environment = env_func()
                        if isinstance(instantiated_environment, list):
                            instantiated_environment = envs_to_dummy_vec_env(instantiated_environment)
                        environments_from_callable.append(instantiated_environment)
                    evaluation_environments = environments_from_callable

                if isinstance(evaluation_environments[0], gym.Env):
                    # noinspection PyCallingNonCallable
                    vectorized_environments = [envs_to_dummy_vec_env(evaluation_environments)]

                if isinstance(evaluation_environments[0], VecEnv):
                    vectorized_environments = evaluation_environments

                def evaluate_agent_on_environment(evaluation_environment: VecEnv, n_episodes: int):
                    n_envs = evaluation_environment.num_envs

                    metric_aggregator = MetricAggregator(connector=connector)
                    log_frequency = min(n_episodes // n_envs, 100)

                    prev_observations = evaluation_environment.reset()
                    prev_actions = [
                        self.choose_action(observation, deterministic=deterministic)
                        for observation in prev_observations
                    ]

                    current_rewards = np.zeros(n_envs)

                    while len(episode_rewards) < n_episodes:
                        observations, rewards, dones, infos = evaluation_environment.step(np.array(prev_actions))
                        metric_aggregator.aggregate_step(observations, prev_actions, rewards, dones, infos)
                        actions = [
                            self.choose_action(observation, deterministic=deterministic) for observation in observations
                        ]

                        current_rewards += rewards

                        prev_actions = actions

                        for i in range(n_envs):
                            if dones[i]:
                                episode_rewards.append(current_rewards[i])
                                pbar.update(1)
                                current_rewards[i] = 0

                                log_episode = (len(metric_aggregator.episode_rewards.get(i, [])) % log_frequency) == 0
                                if log_episode:
                                    metric_aggregator.log_aggregated_metrics(
                                        agent_index=i,
                                        num_timesteps=len(episode_rewards),
                                        log_distributions=False,
                                        metric_name_prefix="Evaluation - ",
                                    )

            threads = []
            for evaluation_environment in vectorized_environments:
                episode_count_per_environment = (n_eval_episodes // len(vectorized_environments)) + 1
                thread = threading.Thread(
                    target=evaluate_agent_on_environment,
                    args=(evaluation_environment, episode_count_per_environment),
                )
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()

        for env in evaluation_environments:
            env.close()

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

    def upload(self, connector: Connector, video_recording_environment: Optional[Environment] = None) -> None:
        """
        Evaluate and upload the decision-making agent to the connector.
            Additional option: Generate a video of the agent interacting with the environment.

        Args:
            connector (Connector): Connector for uploading.
            video_recording_environment (Environment): Environment used for clip creation before upload.
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

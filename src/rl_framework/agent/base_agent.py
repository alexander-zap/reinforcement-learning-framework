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
        self.algorithm_parameters = {**algorithm_parameters} if algorithm_parameters else {}
        self.features_extractor = features_extractor

    def evaluate(
        self,
        evaluation_environments: List[Environment],
        n_eval_episodes: int,
        connector: Connector = DummyConnector(),
        logging_frequency: int = 10,
        deterministic: bool = False,
        close_environments: bool = True,
    ) -> Tuple[float, float]:
        """
        Evaluate the agent for ``n_eval_episodes`` episodes and returns average reward and std of reward.

        Args:
            evaluation_environments (List[Environment]): The evaluation environments.
                For PettingZoo ParallelEnvs, every agent episode counts as one episode. The environment is reset once
                all agents are done. Agents which restart on their own while the other agents continue follow
                gymnasium's auto-reset convention: the done step returns the first observation of the agent's next
                episode (on which its next action is chosen) and passes the last observation of the finished episode
                as `infos[agent]["final_observation"]`.
            n_eval_episodes (int): Number of episodes to evaluate the agent on (at least 1).
                They are split evenly between the environments (and their sub-environments, or PettingZoo agents),
                each contributing a fixed number of episodes, so that environments with shorter episodes are not
                over-represented. Environment factories without episodes to contribute are not instantiated.
            connector (Connector): Connector for logging evaluation metrics.
            logging_frequency (int): Frequency with which evaluation metrics are logged, in episodes of all
                environments.
            deterministic (bool): Whether the agents' actions should be determined in a deterministic or stochastic way.
            close_environments (bool): Whether the given environments are closed after the evaluation (default).
                Environments created from environment factories are always closed.
        """
        if n_eval_episodes < 1:
            raise ValueError(f"n_eval_episodes must be at least 1, got {n_eval_episodes}.")

        def split_evenly(total: int, parts: int) -> List[int]:
            return [(total + index) // parts for index in range(parts)]

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
        # tuple = EnvironmentFactory: its environments are created (and closed) by the evaluation
        environments_from_factories = isinstance(evaluation_environments[0], tuple)

        episode_rewards = []
        episode_rewards_lock = threading.Lock()
        evaluated_episode_count = 0
        # Aggregates the evaluation metrics of (vectorized) gymnasium environments
        metric_aggregator = None

        def count_evaluated_episode() -> int:
            """Count one evaluated episode of any environment and return the running count.
            Used as logging step, which is unique and increasing across all environments (evaluated in parallel).
            """
            nonlocal evaluated_episode_count
            with episode_rewards_lock:
                evaluated_episode_count += 1
                return evaluated_episode_count

        with tqdm(total=n_eval_episodes) as pbar:
            if isinstance(evaluation_environments[0], pettingzoo.ParallelEnv):
                # Each pettingzoo environment is evaluated in its own thread, without vectorization
                vectorized_environments = evaluation_environments

                def evaluate_agent_on_environment(evaluation_environment, n_episodes: int):
                    def choose_actions(observations, rewards, started_agents) -> dict:
                        return {
                            agent: self.choose_action(
                                observations[agent],
                                deterministic=deterministic,
                                reward=0.0 if agent in started_agents else float(rewards.get(agent, 0.0)),
                                episode_start=agent in started_agents,
                                episode_key=(id(evaluation_environment), agent),
                            )
                            for agent in evaluation_environment.agents
                        }

                    prev_observations, _ = evaluation_environment.reset()
                    prev_actions = choose_actions(prev_observations, {}, set(evaluation_environment.agents))
                    running_agents = set(evaluation_environment.agents)

                    episode_reward = {agent: 0.0 for agent in evaluation_environment.agents}
                    local_episode_rewards = []
                    # Each agent contributes a fixed number of episodes (agents appearing later contribute none)
                    possible_agents = list(evaluation_environment.possible_agents)
                    episode_targets = dict(zip(possible_agents, split_evenly(n_episodes, len(possible_agents))))
                    episode_counts = {agent: 0 for agent in possible_agents}

                    while any(episode_counts[agent] < episode_targets[agent] for agent in possible_agents):
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

                        # next action to be executed (based on new observation); an agent starts an episode if it
                        #   appears or restarts (done, but still active: see the docstring)
                        started_agents = {
                            agent
                            for agent in evaluation_environment.agents
                            if agent not in running_agents
                            or terminations.get(agent, False)
                            or truncations.get(agent, False)
                        }
                        actions = choose_actions(observations, rewards, started_agents)
                        running_agents = set(evaluation_environment.agents)

                        for agent in rewards.keys():
                            done = terminations[agent] or truncations[agent]
                            # A done agent without a running episode is either a dead agent still reported (ignore) or
                            #   a restarted agent whose new episode lasted one step (count).
                            #   Only a restarted agent passes the last observation of its finished episode as
                            #   `final_observation` info (see the docstring), which tells the two cases apart.
                            restarted = "final_observation" in infos.get(agent, {})
                            if agent not in episode_reward and (not done or restarted):
                                episode_reward[agent] = rewards[agent]
                            elif agent in episode_reward:
                                episode_reward[agent] += rewards[agent]

                        prev_actions = actions

                        if dones.any():
                            done_indices = np.where(dones == True)[0]
                            for done_index in done_indices:
                                agent = list(terminations.keys())[done_index]
                                if agent in episode_reward:
                                    if episode_counts.get(agent, 0) < episode_targets.get(agent, 0):
                                        local_episode_rewards.append(episode_reward[agent])
                                        episode_counts[agent] += 1
                                        pbar.update(1)
                                    del episode_reward[agent]

                        if env_done:
                            prev_observations, _ = evaluation_environment.reset()
                            prev_actions = choose_actions(prev_observations, {}, set(evaluation_environment.agents))
                            running_agents = set(evaluation_environment.agents)
                            episode_reward = {agent: 0.0 for agent in evaluation_environment.agents}

                    with episode_rewards_lock:
                        episode_rewards.extend(local_episode_rewards)

            elif (
                isinstance(evaluation_environments[0], gym.Env)
                or isinstance(evaluation_environments[0], VecEnv)
                or isinstance(evaluation_environments[0], tuple)
            ):
                # tuple = EnvironmentFactory in format (stub_environment, env_return_function)
                if isinstance(evaluation_environments[0], tuple):
                    environments_from_callable = []
                    # Factories without episodes to contribute (more factories than episodes) are not instantiated
                    factory_episodes = split_evenly(n_eval_episodes, len(evaluation_environments))
                    for (_, env_func), n_factory_episodes in zip(evaluation_environments, factory_episodes):
                        if n_factory_episodes == 0:
                            continue
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

                # One aggregator for all environments (each evaluated in its own thread, hence the lock), so that
                #   every log covers the latest episodes of all of them. Their sub-environments get distinct indices.
                metric_aggregator = MetricAggregator(connector=connector)
                metric_aggregator_lock = threading.Lock()
                log_frequency = max(min(n_eval_episodes, logging_frequency), 1)
                index_offsets, index_offset = {}, 0
                for vectorized_environment in vectorized_environments:
                    index_offsets[id(vectorized_environment)] = index_offset
                    index_offset += vectorized_environment.num_envs

                def log_evaluation_metrics(evaluation_step: int):
                    metric_aggregator.log_and_clear_aggregated_metrics(
                        num_timesteps=evaluation_step, log_distributions=False, metric_name_prefix="Evaluation - "
                    )

                def evaluate_agent_on_environment(evaluation_environment: VecEnv, n_episodes: int):
                    n_envs = evaluation_environment.num_envs

                    def choose_actions(observations, rewards, episode_starts) -> list:
                        return [
                            self.choose_action(
                                observation,
                                deterministic=deterministic,
                                reward=0.0 if episode_starts[index] else float(rewards[index]),
                                episode_start=bool(episode_starts[index]),
                                episode_key=(id(evaluation_environment), index),
                            )
                            for index, observation in enumerate(observations)
                        ]

                    prev_observations = evaluation_environment.reset()
                    prev_actions = choose_actions(prev_observations, np.zeros(n_envs), np.ones(n_envs, dtype=bool))

                    current_rewards = np.zeros(n_envs)
                    local_episode_rewards = []
                    # Each sub-environment contributes a fixed number of episodes
                    episode_targets = np.array(split_evenly(n_episodes, n_envs))
                    episode_counts = np.zeros(n_envs, dtype=int)

                    while (episode_counts < episode_targets).any():
                        observations, rewards, dones, infos = evaluation_environment.step(np.array(prev_actions))
                        # Episodes beyond a sub-environment's quota (while others complete theirs) are not evaluated
                        aggregated_infos = [
                            {**info, "discard": True} if done and count >= target else info
                            for info, done, count, target in zip(infos, dones, episode_counts, episode_targets)
                        ]
                        with metric_aggregator_lock:
                            metric_aggregator.aggregate_step(
                                observations,
                                prev_actions,
                                rewards,
                                dones,
                                aggregated_infos,
                                index_offset=index_offsets[id(evaluation_environment)],
                            )
                        # Sub-environments reset automatically: a done one has started its next episode
                        actions = choose_actions(observations, rewards, dones)

                        current_rewards += rewards

                        prev_actions = actions

                        for i in range(n_envs):
                            if dones[i]:
                                episode_reward = current_rewards[i]
                                current_rewards[i] = 0
                                if episode_counts[i] >= episode_targets[i]:
                                    continue
                                local_episode_rewards.append(episode_reward)
                                episode_counts[i] += 1
                                evaluation_step = count_evaluated_episode()
                                pbar.update(1)

                                with metric_aggregator_lock:
                                    if len(metric_aggregator.episode_rewards) >= log_frequency:
                                        log_evaluation_metrics(evaluation_step)

                    with episode_rewards_lock:
                        episode_rewards.extend(local_episode_rewards)

            else:
                raise TypeError(
                    f"Evaluation is not supported for environments of type {type(evaluation_environments[0])}."
                )

            threads = []
            environment_episodes = split_evenly(n_eval_episodes, len(vectorized_environments))
            for evaluation_environment, n_environment_episodes in zip(vectorized_environments, environment_episodes):
                if n_environment_episodes == 0:
                    continue
                thread = threading.Thread(
                    target=evaluate_agent_on_environment,
                    args=(evaluation_environment, n_environment_episodes),
                )
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()

            # Episodes since the last log (fewer than the logging frequency)
            if metric_aggregator is not None and metric_aggregator.episode_rewards:
                log_evaluation_metrics(evaluated_episode_count)

        if close_environments or environments_from_factories:
            for env in evaluation_environments:
                env.close()

        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        return mean_reward, std_reward

    @abstractmethod
    def choose_action(self, observation: object, deterministic: bool, *args, **kwargs):
        """
        Choose the action for the observation.

        `evaluate` also passes the context of the episode, for agents whose actions depend on it (e.g., d3rlpy's
        decision transformers); other agents ignore it:
            - reward (float): The reward of the episode's previous step (0 at its start).
            - episode_start (bool): Whether the observation is the first of an episode.
            - episode_key (Hashable): Identifies the running episode's environment (and its sub-environment or
              PettingZoo agent); episodes are evaluated interleaved.
        """
        raise NotImplementedError

    @abstractmethod
    def save_policy_as_onnx(self, file_path: Path, *args, **kwargs) -> None:
        """
        Save the agent as an ONNX model.

        Args:
            file_path (Path): Path to save the policy to (in ONNX format).
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

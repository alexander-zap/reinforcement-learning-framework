from collections import Counter, deque
from typing import Any, Deque, List, Union

import numpy as np


class MetricAggregator:
    """
    An object that aggregates metrics across multiple episodes from environment step feedback
        (observations, actions, rewards, terminations, truncations, infos):
        - histogram of performed actions per episode
        - episode rewards
        - mean of any tracked metric (provided as "step_metric_<name>" in the info dict at each step)
        - reason of episode end (provided as "episode_end_reason" in the info dict on terminated step)
    """

    def __init__(self, connector, aggregate_distributions=False):
        self.connector = connector
        self.aggregate_distributions = aggregate_distributions

        # Tracking episode reward per episode
        #   (np.array, one index per agent, continuously updated by adding rewards at each step)
        self.episode_reward: Union[np.ndarray | None] = None
        self.episode_rewards: dict[int, List[float]] = {}
        # Tracking actions per episode (one list per agent, updated by appending action at each step)
        # NOTE: Even though officially actions can be Any, for logging we assume they are int or float
        self.episode_actions: Union[List[List[Any]] | None] = None
        # Tracking other numeric metrics for each step episode (name -> value)
        self.episode_step_metrics: dict[int, List[List[float]]] = {}

        # Tracking stats across multiple episodes
        self.episode_end_reasons: dict[int, Deque] = {}

    def aggregate_step(self, observations, actions, rewards, dones, infos):
        """
        This method will be called by the model after each call to `env.step()`.
        If the callback returns False, training is aborted early.
        """
        # Rewards
        if self.episode_reward is None:
            self.episode_reward = rewards
        else:
            self.episode_reward += rewards

        # Actions
        if self.aggregate_distributions:
            if not self.episode_actions:
                self.episode_actions = [[action] for action in actions]
            else:
                for agent_index, action in enumerate(actions):
                    self.episode_actions[agent_index].append(action)

        # Infos (step metrics)
        for agent_index, info in enumerate(infos):
            for key, value in info.items():
                if key.startswith("step_metric_"):
                    metric_name = key[len("step_metric_") :]
                    if metric_name not in self.episode_step_metrics:
                        self.episode_step_metrics[metric_name] = [[] for _ in range(len(infos))]
                    self.episode_step_metrics[metric_name][agent_index].append(float(value))

        # Episode end reasons
        done_indices = np.where(dones == True)[0]
        if done_indices.size != 0:
            for done_index in done_indices:
                if not infos[agent_index].get("discard", False):
                    if done_index not in self.episode_rewards:
                        self.episode_rewards[done_index] = []
                    self.episode_rewards[done_index].append(self.episode_reward[done_index])
                    self.episode_reward[done_index] = 0

                    if infos[done_index].get("episode_end_reason", None) is not None:
                        if done_index not in self.episode_end_reasons:
                            self.episode_end_reasons[done_index] = deque(maxlen=100)
                        self.episode_end_reasons[done_index].append(infos[done_index]["episode_end_reason"])

    def log_aggregated_metrics(self, agent_index, num_timesteps, log_distributions=False, metric_name_prefix=""):
        self.connector.log_value_with_timestep(
            num_timesteps, np.mean(self.episode_rewards[agent_index]), f"{metric_name_prefix}Episode reward"
        )
        # Log other tracked step metrics
        for metric_name, per_agent_values in self.episode_step_metrics.items():
            if per_agent_values[agent_index]:
                self.connector.log_value_with_timestep(
                    num_timesteps,
                    np.mean(per_agent_values[agent_index]),
                    value_name=f"Mean - {metric_name}",
                    title_name=f"{metric_name_prefix}{metric_name}",
                )
                self.connector.log_value_with_timestep(
                    num_timesteps,
                    np.std(per_agent_values[agent_index]),
                    value_name=f"Std - {metric_name}",
                    title_name=f"{metric_name_prefix}{metric_name}",
                )
                self.connector.log_value_with_timestep(
                    num_timesteps,
                    np.max(per_agent_values[agent_index]),
                    value_name=f"Max - {metric_name}",
                    title_name=f"{metric_name_prefix}{metric_name}",
                )
                self.connector.log_value_with_timestep(
                    num_timesteps,
                    np.min(per_agent_values[agent_index]),
                    value_name=f"Min - {metric_name}",
                    title_name=f"{metric_name_prefix}{metric_name}",
                )

        if agent_index in self.episode_end_reasons and self.episode_end_reasons[agent_index]:
            counter = Counter(self.episode_end_reasons[agent_index])
            for reason, count in counter.items():
                self.connector.log_value_with_timestep(
                    num_timesteps,
                    count / len(self.episode_end_reasons[agent_index]),
                    value_name=f"Episode end reason - {reason}",
                    title_name=f"{metric_name_prefix}Episode end reasons",
                )

        if log_distributions:
            # Log action distribution
            if self.episode_actions[agent_index]:
                if isinstance(self.episode_actions[agent_index][0], np.ndarray):
                    for action_index, action_sequence in enumerate(zip(*self.episode_actions[agent_index])):
                        self.connector.log_histogram_with_timestep(
                            num_timesteps,
                            np.array(action_sequence),
                            f"{metric_name_prefix}Action distribution - action dim {action_index}",
                        )
                else:
                    self.connector.log_histogram_with_timestep(
                        num_timesteps,
                        np.array(self.episode_actions[agent_index]),
                        f"{metric_name_prefix}Action distribution",
                    )

            # Log other tracked step metric distributions
            for metric_name, per_agent_values in self.episode_step_metrics.items():
                if per_agent_values[agent_index]:
                    self.connector.log_histogram_with_timestep(
                        num_timesteps,
                        np.array(per_agent_values[agent_index]),
                        f"{metric_name_prefix}Distribution - {metric_name}",
                    )

    def reset_multi_episode_trackers(self, agent_index: int):
        self.episode_rewards[agent_index] = []
        for metric_name in self.episode_step_metrics.keys():
            self.episode_step_metrics[metric_name][agent_index] = []
        if self.episode_actions:
            self.episode_actions[agent_index] = []

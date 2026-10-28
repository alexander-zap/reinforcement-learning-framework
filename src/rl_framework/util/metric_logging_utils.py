import math
from collections import Counter
from typing import Any, Dict, List

import numpy as np


class RunningStatistics:
    """Count, mean, (population) standard deviation, minimum and maximum of values merged in batches, without storing
    the values."""

    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.sum_squared_deviations = 0.0
        self.min = math.inf
        self.max = -math.inf

    def merge(self, values: List[float]) -> None:
        values = np.asarray(values, dtype=float)
        batch_mean = values.mean()
        delta = batch_mean - self.mean
        count = self.count + values.size
        squared_deviations = ((values - batch_mean) ** 2).sum()
        self.sum_squared_deviations += squared_deviations + delta**2 * self.count * values.size / count
        self.mean += delta * values.size / count
        self.count = count
        self.min = float(np.minimum(self.min, values.min()))
        self.max = float(np.maximum(self.max, values.max()))

    @property
    def std(self) -> float:
        return math.sqrt(self.sum_squared_deviations / self.count)


class MetricAggregator:
    """
    An object that aggregates metrics over all episodes finished since the last log, from environment step feedback
        (observations, actions, rewards, terminations, truncations, infos):
        - histogram of performed actions
        - episode rewards
        - mean of any tracked metric (provided as "step_metric_<name>" in the info dict at each step)
        - reason of episode end (provided as "episode_end_reason" in the info dict on terminated step)
        - share of dummy steps among all steps (provided as "dummy_step" in the info dict, e.g., the steps of PettingZoo
          agents which left before the others), once a dummy step occurred

    Episodes running at the same time (one per agent index, e.g., per environment of a vectorized environment) are
    tracked separately until they end. Finished episodes are pooled; discarded ones are dropped.
    """

    def __init__(self, connector, aggregate_distributions=False):
        self.connector = connector
        self.aggregate_distributions = aggregate_distributions

        # Running episodes, by agent index
        self.running_episode_reward: Dict[int, float] = {}
        # NOTE: Even though officially actions can be Any, for logging we assume they are int or float
        self.running_episode_actions: Dict[int, List[Any]] = {}
        self.running_episode_step_metrics: Dict[int, Dict[str, List[float]]] = {}

        # Finished episodes since the last log (step metrics as statistics, not values, to bound memory)
        self.episode_rewards: List[float] = []
        self.episode_end_reasons: Counter = Counter()
        self.step_metric_statistics: Dict[str, RunningStatistics] = {}

        # Finished episodes since the last log, raw values only for distributions
        self.distribution_actions: List[Any] = []
        self.distribution_step_metric_values: Dict[str, List[float]] = {}

        # Every episode end reason seen so far, logged with share 0 when absent since the last log
        self.seen_end_reasons: set = set()

        # Steps (of all agent indices) and dummy steps since the last log; logged once a dummy step occurred
        self.step_count = 0
        self.dummy_step_count = 0
        self.seen_dummy_steps = False

    def aggregate_step(self, observations, actions, rewards, dones, infos, index_offset: int = 0):
        """
        This method will be called by the model after each call to `env.step()`.

        Args:
            index_offset: Offset of the agent indices, for steps of different environments into the same aggregator.
        """
        for index, info in enumerate(infos):
            agent_index = index_offset + index

            self.step_count += 1
            if info.get("dummy_step", False):
                self.dummy_step_count += 1
                self.seen_dummy_steps = True

            running_reward = self.running_episode_reward.get(agent_index, 0.0)
            self.running_episode_reward[agent_index] = running_reward + rewards[index]

            if self.aggregate_distributions:
                self.running_episode_actions.setdefault(agent_index, []).append(actions[index])

            step_metrics = self.running_episode_step_metrics.setdefault(agent_index, {})
            for key, value in info.items():
                if key.startswith("step_metric_"):
                    metric_name = key[len("step_metric_") :]
                    values = step_metrics.get(metric_name)
                    if values is None:
                        values = step_metrics[metric_name] = []
                    values.append(float(value))
            if dones[index]:
                self._finish_episode(agent_index, info)

    def _finish_episode(self, agent_index: int, info: dict) -> None:
        running_episode_reward = self.running_episode_reward.pop(agent_index)
        actions = self.running_episode_actions.pop(agent_index, [])
        step_metrics = self.running_episode_step_metrics.pop(agent_index, {})
        if info.get("discard", False):
            return

        self.episode_rewards.append(running_episode_reward)
        for metric_name, values in step_metrics.items():
            if metric_name not in self.step_metric_statistics:
                self.step_metric_statistics[metric_name] = RunningStatistics()
            self.step_metric_statistics[metric_name].merge(values)
            if self.aggregate_distributions:
                self.distribution_step_metric_values.setdefault(metric_name, []).extend(values)
        if self.aggregate_distributions:
            self.distribution_actions.extend(actions)
        if (end_reason := info.get("episode_end_reason", None)) is not None:
            self.episode_end_reasons[end_reason] += 1
            self.seen_end_reasons.add(end_reason)

    def log_and_clear_aggregated_metrics(self, num_timesteps, log_distributions=False, metric_name_prefix=""):
        """Log the metrics of the episodes finished since the last log, then clear them.
        Running episodes are kept, they belong to the next log."""
        # No recorded (non-discarded) episode since the last log: nothing to average
        if self.episode_rewards:
            self.connector.log_value_with_timestep(
                num_timesteps, np.mean(self.episode_rewards), f"{metric_name_prefix}Episode reward"
            )
        # Log other tracked step metrics
        for metric_name, statistics in self.step_metric_statistics.items():
            if statistics.count:
                self.connector.log_value_with_timestep(
                    num_timesteps,
                    statistics.mean,
                    value_name=f"Mean - {metric_name}",
                    title_name=f"{metric_name_prefix}{metric_name}",
                )
                self.connector.log_value_with_timestep(
                    num_timesteps,
                    statistics.std,
                    value_name=f"Std - {metric_name}",
                    title_name=f"{metric_name_prefix}{metric_name}",
                )
                self.connector.log_value_with_timestep(
                    num_timesteps,
                    statistics.max,
                    value_name=f"Max - {metric_name}",
                    title_name=f"{metric_name_prefix}{metric_name}",
                )
                self.connector.log_value_with_timestep(
                    num_timesteps,
                    statistics.min,
                    value_name=f"Min - {metric_name}",
                    title_name=f"{metric_name_prefix}{metric_name}",
                )

        if self.episode_end_reasons:
            episode_count = sum(self.episode_end_reasons.values())
            # Absent reasons are logged as 0, so that averaging over the series does not overstate rare reasons
            for reason in self.seen_end_reasons:
                self.connector.log_value_with_timestep(
                    num_timesteps,
                    self.episode_end_reasons[reason] / episode_count,
                    value_name=f"Episode end reason - {reason}",
                    title_name=f"{metric_name_prefix}Episode end reasons",
                )

        if self.seen_dummy_steps and self.step_count:
            self.connector.log_value_with_timestep(
                num_timesteps, self.dummy_step_count / self.step_count, f"{metric_name_prefix}Dummy step share"
            )

        if log_distributions:
            # Log action distribution
            if self.distribution_actions:
                if isinstance(self.distribution_actions[0], np.ndarray):
                    for action_index, action_sequence in enumerate(zip(*self.distribution_actions)):
                        self.connector.log_histogram_with_timestep(
                            num_timesteps,
                            np.array(action_sequence),
                            f"{metric_name_prefix}Action distribution - action dim {action_index}",
                        )
                else:
                    self.connector.log_histogram_with_timestep(
                        num_timesteps,
                        np.array(self.distribution_actions),
                        f"{metric_name_prefix}Action distribution",
                    )

            # Log other tracked step metric distributions
            for metric_name, values in self.distribution_step_metric_values.items():
                if values:
                    self.connector.log_histogram_with_timestep(
                        num_timesteps,
                        np.array(values),
                        f"{metric_name_prefix}Distribution - {metric_name}",
                    )

        self.episode_rewards = []
        self.episode_end_reasons = Counter()
        self.step_count = 0
        self.dummy_step_count = 0
        self.step_metric_statistics = {}
        self.distribution_actions = []
        self.distribution_step_metric_values = {}

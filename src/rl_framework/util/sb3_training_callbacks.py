from collections import deque
from typing import Deque, Union

import numpy as np
from async_gym_agents import constants
from async_gym_agents.callback_batching import (
    resolve_episode_action_field,
    resolve_episode_reward_field,
)
from async_gym_agents.data_classes import EpisodeCallbackContext
from async_gym_agents.episode_codec import (
    get_episode_infos,
    get_episode_reset_infos,
    slice_episode_field,
)
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

from .metric_logging_utils import MetricAggregator


def add_callbacks_to_callback(callbacks_to_add: CallbackList, callback_to_be_added_to: BaseCallback):
    if callback_to_be_added_to is None:
        callback_to_be_added_to = CallbackList([])
    elif not isinstance(callback_to_be_added_to, CallbackList):
        callback_to_be_added_to = CallbackList([callback_to_be_added_to])

    for callback in callbacks_to_add.callbacks:
        if callback not in callback_to_be_added_to.callbacks:
            callback_to_be_added_to.callbacks.append(callback)


class EpisodeBatchableCallbackMixin:
    """Advance SB3 bookkeeping without invoking the per-step hook.

    Satisfies half of async_gym_agents.callback_batching.EpisodeBatchableCallback;
    each callback below implements the other half, `process_episode`, so that
    async-gym-agents can process a complete episode instead of every transition.
    """

    def advance_callback(self, transition_count: int, num_timesteps: int) -> None:
        self.n_calls += transition_count
        self.num_timesteps = num_timesteps


class LoggingCallback(EpisodeBatchableCallbackMixin, BaseCallback):
    """
    A custom callback that logs after every done episode:
        - histogram of performed actions per episode
        - episode rewards
        - mean of any tracked metric (provided as "step_metric_<name>" in the info dict at each step)
        - reason of episode end (provided as "episode_end_reason" in the info dict on terminated step)
    """

    def __init__(self, connector, logging_frequency=1, log_distributions=False, verbose=0):
        """
        Args:
            verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
        """
        super().__init__(verbose)
        self.connector = connector
        self.logging_frequency = logging_frequency
        self.log_distributions = log_distributions
        self.episode_counter: dict[str, int] = {}
        self.metric_aggregator = MetricAggregator(connector=connector, aggregate_distributions=log_distributions)
        self.logged_metadata_by_key: dict[str, object] = {}

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.
        If the callback returns False, training is aborted early.
        """
        # Update trackers
        self.metric_aggregator.aggregate_step(
            self.locals["new_obs"],
            self.locals["actions"],
            self.locals["rewards"],
            self.locals["dones"],
            self.locals["infos"],
        )

        # Log meta infos
        for agent_index, info in enumerate(self.locals["infos"]):
            for key, value in info.items():
                if key.startswith("meta_"):
                    if isinstance(value, dict):
                        self.connector.log_dict(value, key)
                    else:
                        self.connector.log_dict({key: value}, key)

        # Log metrics at end of episode
        done_indices = np.where(self.locals["dones"] == True)[0]
        if done_indices.size != 0:
            for done_index in done_indices:
                self.episode_counter[done_index] = self.episode_counter.get(done_index, 0) + 1
                log_this_episode = self.episode_counter[done_index] % self.logging_frequency == 0

                if log_this_episode:
                    self.metric_aggregator.log_aggregated_metrics(
                        agent_index=done_index,
                        num_timesteps=self.num_timesteps,
                        log_distributions=self.log_distributions,
                    )
                    self.metric_aggregator.reset_multi_episode_trackers(done_index)

        return True

    def process_episode(self, context: EpisodeCallbackContext) -> bool:
        """Aggregate framework metrics directly from a complete episode batch."""
        batch = context.batch
        if self._supports_episode_aggregation(self.metric_aggregator):
            self._aggregate_episode(batch)
        else:
            self._aggregate_episode_by_step(batch)
            self._log_episode_metadata(batch)

        terminal_dones = slice_episode_field(
            batch,
            "dones",
            batch.transition_count - 1,
        )
        for done_index in np.flatnonzero(terminal_dones):
            self.episode_counter[done_index] = self.episode_counter.get(done_index, 0) + 1
            if self.episode_counter[done_index] % self.logging_frequency == 0:
                self.metric_aggregator.log_aggregated_metrics(
                    agent_index=done_index,
                    num_timesteps=context.end_timestep,
                    log_distributions=self.log_distributions,
                )
                self.metric_aggregator.reset_multi_episode_trackers(done_index)

        return True

    def _aggregate_episode_by_step(self, batch) -> None:
        for transition_index in range(batch.transition_count):
            infos = get_episode_infos(batch, transition_index)
            self.metric_aggregator.aggregate_step(
                slice_episode_field(batch, "new_obs", transition_index),
                slice_episode_field(
                    batch,
                    resolve_episode_action_field(batch.episode_kind),
                    transition_index,
                ),
                slice_episode_field(
                    batch,
                    resolve_episode_reward_field(batch.episode_kind),
                    transition_index,
                ),
                slice_episode_field(batch, "dones", transition_index),
                infos,
            )

    def _aggregate_episode(self, batch) -> None:
        aggregator = self.metric_aggregator
        # Each complete environment is packed into its own episode batch.
        episode_agent_index = 0
        terminal_index = batch.transition_count - 1
        terminal_dones = slice_episode_field(batch, "dones", terminal_index)
        terminal_infos = get_episode_infos(batch, terminal_index)
        discarded = any(
            terminal_infos[done_index].get(constants.DISCARD_INFO_KEY, False)
            for done_index in np.flatnonzero(terminal_dones)
        )
        if discarded:
            self._log_episode_metadata(batch)
            if aggregator.episode_reward is None:
                reward_values = batch.fields[resolve_episode_reward_field(batch.episode_kind)]
                aggregator.episode_reward = np.zeros(1, dtype=reward_values.dtype)
            aggregator.episode_reward[episode_agent_index] = 0
            return

        rewards = batch.fields[resolve_episode_reward_field(batch.episode_kind)]
        if aggregator.episode_reward is None:
            aggregator.episode_reward = np.zeros(1, dtype=rewards.dtype)
        aggregator.episode_reward[episode_agent_index] += np.sum(rewards)

        if aggregator.aggregate_distributions:
            if not aggregator.episode_actions:
                aggregator.episode_actions = [[]]
            action_field = resolve_episode_action_field(batch.episode_kind)
            aggregator.episode_actions[episode_agent_index].extend(batch.fields[action_field])

        self._aggregate_episode_infos(batch)
        for done_index in np.flatnonzero(terminal_dones):
            terminal_info = terminal_infos[done_index]
            aggregator.episode_rewards.setdefault(done_index, []).append(aggregator.episode_reward[done_index])
            end_reason = terminal_info.get(constants.EPISODE_END_REASON_INFO_KEY)
            if end_reason is not None:
                aggregator.episode_end_reasons.setdefault(
                    done_index,
                    deque(maxlen=constants.EPISODE_END_REASON_WINDOW_SIZE),
                ).append(end_reason)
            aggregator.episode_reward[done_index] = 0

    def _aggregate_episode_infos(self, batch) -> None:
        aggregator = self.metric_aggregator
        episode_step_metrics = aggregator.episode_step_metrics
        step_metric_prefix = constants.STEP_METRIC_INFO_PREFIX
        meta_info_prefix = constants.META_INFO_PREFIX
        for infos in batch.infos.values():
            for agent_index, info in enumerate(infos):
                for key, value in info.items():
                    if key.startswith(step_metric_prefix):
                        metric_name = key[len(step_metric_prefix) :]
                        per_agent_values = episode_step_metrics.get(metric_name)
                        if per_agent_values is None:
                            per_agent_values = [[] for _ in range(len(infos))]
                            episode_step_metrics[metric_name] = per_agent_values
                        per_agent_values[agent_index].append(float(value))
                    elif key.startswith(meta_info_prefix):
                        self._log_metadata(key, value)

    def _log_episode_metadata(self, batch) -> None:
        for infos in batch.infos.values():
            for info in infos:
                for key, value in info.items():
                    if key.startswith(constants.META_INFO_PREFIX):
                        self._log_metadata(key, value)

    def _log_metadata(self, key: str, value: object) -> None:
        if key in self.logged_metadata_by_key and self._metadata_matches(
            self.logged_metadata_by_key[key],
            value,
        ):
            return

        if isinstance(value, dict):
            self.connector.log_dict(value, key)
        else:
            self.connector.log_dict({key: value}, key)
        self.logged_metadata_by_key[key] = value

    @staticmethod
    def _metadata_matches(previous_value: object, current_value: object) -> bool:
        try:
            comparison = previous_value == current_value
            if isinstance(comparison, np.ndarray):
                return bool(np.all(comparison))
            return bool(comparison)
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _supports_episode_aggregation(metric_aggregator: object) -> bool:
        if type(metric_aggregator).__name__ != "MetricAggregator":
            return False
        return all(
            hasattr(metric_aggregator, name)
            for name in (
                "aggregate_distributions",
                "episode_actions",
                "episode_end_reasons",
                "episode_reward",
                "episode_rewards",
                "episode_step_metrics",
            )
        )


class SavingCallback(EpisodeBatchableCallbackMixin, BaseCallback):
    """
    A custom callback which uploads the agent to the connector after every `checkpoint_frequency` steps.
    """

    def __init__(self, agent, connector, checkpoint_frequency=50000, verbose=0):
        """
        Args:
            checkpoint_frequency: After how many steps a checkpoint should be saved to the connector.
            verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
        """
        super().__init__(verbose)
        self.agent = agent
        self.connector = connector
        self.checkpoint_frequency = checkpoint_frequency
        self.next_upload = checkpoint_frequency

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.
        If the callback returns False, training is aborted early.
        """
        if self.num_timesteps > self.next_upload:
            self.connector.upload(
                agent=self.agent,
                checkpoint_id=self.num_timesteps,
            )
            self.next_upload = self.num_timesteps + self.checkpoint_frequency

        return True

    def process_episode(self, context: EpisodeCallbackContext) -> bool:
        """Preserve checkpoint scheduling without checking it every transition.

        Batched by episode for both on- and off-policy collection: every
        checkpoint threshold crossed within the batch is replayed in order,
        so a long-running off-policy episode still uploads at the same
        cumulative-timestep boundaries a per-step check would have used.
        """
        checkpoint_timestep = max(
            context.start_timestep + 1,
            self.next_upload + 1,
        )
        while checkpoint_timestep <= context.end_timestep:
            self.num_timesteps = checkpoint_timestep
            self.connector.upload(
                agent=self.agent,
                checkpoint_id=checkpoint_timestep,
            )
            self.next_upload = checkpoint_timestep + self.checkpoint_frequency
            checkpoint_timestep = max(
                checkpoint_timestep + 1,
                self.next_upload + 1,
            )

        return True


class ExperimentPruningCallback(EpisodeBatchableCallbackMixin, BaseCallback):
    """
    A custom callback which stop the training the experiment does not reach performance threshold at given step amount.
    """

    def __init__(self, episode_reward_threshold: float = 0.0, pruning_start_at: int = 1000000, verbose=0):
        """
        Args:
            episode_reward_threshold: If the mean episode reward of the last 1000 episodes is below this threshold
                (after `pruning_start_at` steps), the experiment will be pruned.
            pruning_start_at: After how many steps the potential pruning should start.
            verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
        """
        super().__init__(verbose)

        self.episode_reward_threshold = episode_reward_threshold
        self.pruning_start_at = pruning_start_at

        # Continuously tracking episode reward for all agents
        #   (np.array, one index per agent, continuously updated by adding rewards at each step)
        self.accumulation_of_episode_reward: Union[np.ndarray | None] = None
        # Saving episode rewards of last 1000 episodes (for all agents)
        self.episode_rewards: Deque[float] = deque(maxlen=1000)

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.
        If the callback returns False, training is aborted early.
        """
        if self.episode_reward is None:
            self.episode_reward = self.locals["rewards"]
        else:
            self.episode_reward += self.locals["rewards"]

        done_indices = np.where(self.locals["dones"] == True)[0]
        if done_indices.size != 0:
            for done_index in done_indices:
                if not self.locals["infos"][done_index].get("discard", False):
                    self.episode_rewards.append(self.episode_reward[done_index])
                # Reset trackers for done agent
                self.episode_reward[done_index] = 0

        if self.num_timesteps > self.pruning_start_at and len(self.episode_rewards) == self.episode_rewards.maxlen:
            mean_episode_reward = np.mean(self.episode_rewards)
            if mean_episode_reward < self.episode_reward_threshold:
                return False

        return True

    def process_episode(self, context: EpisodeCallbackContext) -> bool:
        """Evaluate pruning once when a complete episode changes its reward window."""
        batch = context.batch
        episode_rewards = np.sum(
            batch.fields[resolve_episode_reward_field(batch.episode_kind)],
            axis=0,
            keepdims=True,
        )
        terminal_index = batch.transition_count - 1
        terminal_dones = slice_episode_field(batch, "dones", terminal_index)
        terminal_infos = get_episode_infos(batch, terminal_index)

        for done_index in np.flatnonzero(terminal_dones):
            if not terminal_infos[done_index].get(constants.DISCARD_INFO_KEY, False):
                self.episode_rewards.append(episode_rewards[done_index])

        self.episode_reward = np.zeros_like(episode_rewards)
        reward_window_is_full = len(self.episode_rewards) == self.episode_rewards.maxlen
        if context.end_timestep <= self.pruning_start_at or not reward_window_is_full:
            return True
        return bool(np.mean(self.episode_rewards) >= self.episode_reward_threshold)


class ResetInfoCallback(EpisodeBatchableCallbackMixin, BaseCallback):
    """
    A custom callback that logs after every reset the reset_infos dict..
    """

    def __init__(self, connector, verbose=0):
        """
        Args:
            verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
        """
        super().__init__(verbose)
        self.connector = connector
        # Tracking episode number for each agent (agent index -> episode count)
        self.episode_counter: dict[str, int] = {}
        # Tracking agents which have done a first step (list of agent indices)
        self.first_step_tracker: list[str] = []

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.
        If the callback returns False, training is aborted early.
        """
        if not self.locals.get("reset_infos", None):
            self.update_locals({"reset_infos": self.training_env.reset_infos})

        # Write reset info at first step of first episode
        for agent_index, reset_info in enumerate(self.locals["reset_infos"]):
            if agent_index not in self.first_step_tracker:
                self.episode_counter[agent_index] = 0
                self.first_step_tracker.append(agent_index)
                if reset_info:
                    self.connector.log_dict(
                        reset_info,
                        f"Reset Info - Agent {agent_index} - Episode {self.episode_counter.get(agent_index, 0)}",
                    )

        # Write reset info at each reset (when done=True, reset has already happened and reset_info is available)
        done_indices = np.where(self.locals["dones"] == True)[0]
        if done_indices.size != 0:
            for done_index in done_indices:
                self.episode_counter[done_index] = self.episode_counter.get(done_index, 0) + 1
                reset_info = self.locals["reset_infos"][done_index]
                if reset_info:
                    self.connector.log_dict(
                        reset_info,
                        f"Reset Info - Agent {done_index} - Episode {self.episode_counter.get(done_index, 0)}",
                    )

        return True

    def process_episode(self, context: EpisodeCallbackContext) -> bool:
        """Log initial and post-terminal reset information once per episode."""
        batch = context.batch
        initial_reset_infos = get_episode_reset_infos(batch, 0)
        for agent_index, reset_info in enumerate(initial_reset_infos):
            if agent_index in self.first_step_tracker:
                continue
            self.episode_counter[agent_index] = 0
            self.first_step_tracker.append(agent_index)
            self._log_reset_info(agent_index, reset_info)

        terminal_index = batch.transition_count - 1
        terminal_dones = slice_episode_field(batch, "dones", terminal_index)
        terminal_reset_infos = get_episode_reset_infos(batch, terminal_index)
        for done_index in np.flatnonzero(terminal_dones):
            self.episode_counter[done_index] = self.episode_counter.get(done_index, 0) + 1
            self._log_reset_info(done_index, terminal_reset_infos[done_index])

        return True

    def _log_reset_info(self, agent_index: int, reset_info: dict) -> None:
        if not reset_info:
            return
        self.connector.log_dict(
            reset_info,
            f"Reset Info - Agent {agent_index} - Episode {self.episode_counter[agent_index]}",
        )

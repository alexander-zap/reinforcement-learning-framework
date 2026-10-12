from collections import deque
from typing import Callable, Deque, List, Optional, Union

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


def add_callbacks_to_callback(
    callbacks_to_add: CallbackList,
    callback_to_be_added_to: Union[BaseCallback, List[BaseCallback], None],
) -> CallbackList:
    """
    Combine callbacks without modifying the arguments.

    Args:
        callbacks_to_add: Callbacks to append.
        callback_to_be_added_to: None, a single callback, a list of callbacks or a CallbackList.

    Returns:
        A new CallbackList with the callbacks of `callback_to_be_added_to`, followed by those of `callbacks_to_add`
            which it does not contain yet. It has to be used in place of `callback_to_be_added_to`.
    """
    if callback_to_be_added_to is None:
        existing_callbacks = []
    elif isinstance(callback_to_be_added_to, CallbackList):
        existing_callbacks = list(callback_to_be_added_to.callbacks)
    elif isinstance(callback_to_be_added_to, list):
        existing_callbacks = list(callback_to_be_added_to)
    else:
        existing_callbacks = [callback_to_be_added_to]

    new_callbacks = [callback for callback in callbacks_to_add.callbacks if callback not in existing_callbacks]
    return CallbackList(existing_callbacks + new_callbacks)


class EpisodeBatchableCallbackMixin:
    """Advance SB3 bookkeeping without invoking the per-step hook.

    Satisfies half of async_gym_agents.callback_batching.EpisodeBatchableCallback;
    each callback below implements the other half, `process_episode`, so that
    async-gym-agents can process a complete episode instead of every transition.
    """

    def advance_callback(self, transition_count: int, num_timesteps: int) -> None:
        self.n_calls += transition_count
        self.num_timesteps = num_timesteps

    def _get_agent_index(self, context: EpisodeCallbackContext) -> int:
        """
        Agent index of an episode's origin, i.e., its worker and the sub-environment of the worker's (vectorized)
        environment. async-gym-agents batches every episode as a single environment, so the batch's own index is
        always 0. Origins are numbered in order of first appearance.
        """
        origin = (context.worker_index, context.env_index)
        agent_indices = self.__dict__.setdefault("_agent_indices", {})
        return agent_indices.setdefault(origin, len(agent_indices))


class LoggingCallback(EpisodeBatchableCallbackMixin, BaseCallback):
    """
    A custom callback that logs after every done episode:
        - histogram of performed actions per episode
        - episode rewards
        - mean of any tracked metric (provided as "step_metric_<name>" in the info dict at each step)
        - reason of episode end (provided as "episode_end_reason" in the info dict on terminated step)

    Metrics are aggregated per agent: per environment of the vectorized environment (SB3) or per episode origin,
    i.e., worker and sub-environment (async-gym-agents). Each agent is logged after every `logging_frequency` of
    its own completed episodes, into the same series as all other agents.
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
        # Episode-batched mode (async-gym-agents): one aggregator per agent, since each batch is a single environment
        self.episode_metric_aggregators: dict[int, MetricAggregator] = {}
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
        for info in self.locals["infos"]:
            self._log_meta_infos(info)

        # Log metrics at end of episode
        done_indices = np.where(self.locals["dones"] == True)[0]
        for done_index in done_indices:
            self._log_if_due(done_index, self.num_timesteps, self.locals["infos"][done_index])

        return True

    def process_episode(self, context: EpisodeCallbackContext) -> bool:
        """Aggregate framework metrics one transition at a time, exactly as `_on_step` does.

        Unlike checkpoint/pruning/reset-info handling below, this can't be reduced to
        inspecting only the episode's boundary transitions: `aggregate_step` needs every
        transition's reward/action/step-metrics, since any of them may carry a metric.
        """
        batch = context.batch
        agent_index = self._get_agent_index(context)
        metric_aggregator = self.episode_metric_aggregators.setdefault(
            agent_index,
            MetricAggregator(connector=self.connector, aggregate_distributions=self.log_distributions),
        )
        for transition_index in range(batch.transition_count):
            self._process_transition(batch, transition_index, metric_aggregator)

        terminal_index = batch.transition_count - 1
        terminal_dones = slice_episode_field(batch, "dones", terminal_index)
        terminal_infos = get_episode_infos(batch, terminal_index)
        for done_index in np.flatnonzero(terminal_dones):
            self._log_if_due(
                agent_index, context.end_timestep, terminal_infos[done_index], metric_aggregator, done_index
            )

        return True

    def _process_transition(self, batch, transition_index: int, metric_aggregator: MetricAggregator) -> None:
        infos = get_episode_infos(batch, transition_index)
        metric_aggregator.aggregate_step(
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
        for info in infos:
            self._log_meta_infos(info)

    def _log_meta_infos(self, info: dict) -> None:
        """Log every `meta_*` entry in one step's/transition's info dict, deduping unchanged values."""
        for key, value in info.items():
            if key.startswith(constants.META_INFO_PREFIX):
                self._log_metadata(key, value)

    def _log_if_due(
        self,
        agent_index: int,
        num_timesteps: int,
        info: dict,
        metric_aggregator: Optional[MetricAggregator] = None,
        aggregator_index: Optional[int] = None,
    ) -> None:
        """Log and reset one agent's aggregated metrics if its logging_frequency is met.

        Shared by both `_on_step` (one done agent at a time, in the shared aggregator at index `agent_index`) and
        `process_episode` (the agent's own aggregator, at the batch's index `aggregator_index`).
        Discarded episodes are not recorded by the metric aggregator, so they do not count towards logging.
        """
        if info.get(constants.DISCARD_INFO_KEY, False):
            return
        metric_aggregator = metric_aggregator or self.metric_aggregator
        aggregator_index = agent_index if aggregator_index is None else aggregator_index
        self.episode_counter[agent_index] = self.episode_counter.get(agent_index, 0) + 1
        if self.episode_counter[agent_index] % self.logging_frequency == 0:
            metric_aggregator.log_aggregated_metrics(
                agent_index=aggregator_index,
                num_timesteps=num_timesteps,
                log_distributions=self.log_distributions,
            )
            metric_aggregator.reset_multi_episode_trackers(aggregator_index)

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
        self._maybe_upload(self.num_timesteps)
        return True

    def process_episode(self, context: EpisodeCallbackContext) -> bool:
        """Preserve checkpoint scheduling without checking it every transition.

        Uploads are keyed off elapsed timesteps, not `done`: the first checkpoint boundary crossed within the
        episode uploads (named after that timestep, as in `_on_step`). The model does not change while an episode
        batch is processed, so further boundaries crossed by the same (long) episode are skipped instead of
        uploading the same model again; the schedule keeps its `checkpoint_frequency` rhythm.
        """
        checkpoint_timestep = max(context.start_timestep + 1, self.next_upload + 1)
        if checkpoint_timestep <= context.end_timestep:
            self._maybe_upload(checkpoint_timestep)
            while self.next_upload < context.end_timestep:
                self.next_upload += self.checkpoint_frequency

        return True

    def _maybe_upload(self, timestep: int) -> None:
        """Upload a checkpoint if `timestep` has crossed the next scheduled upload."""
        if timestep > self.next_upload:
            self.num_timesteps = timestep
            self.connector.upload(
                agent=self.agent,
                checkpoint_id=timestep,
            )
            self.next_upload = timestep + self.checkpoint_frequency


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
        self.episode_reward: Union[np.ndarray | None] = None
        # Saving episode rewards of last 1000 episodes (for all agents)
        self.episode_rewards: Deque[float] = deque(maxlen=1000)

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.
        If the callback returns False, training is aborted early.
        """
        if self.episode_reward is None:
            self.episode_reward = np.zeros_like(self.locals["rewards"])
        self.episode_reward += self.locals["rewards"]

        for done_index in np.where(self.locals["dones"] == True)[0]:
            self._record_completed_episode(self.episode_reward[done_index], self.locals["infos"][done_index])
            # Reset tracker for done agent
            self.episode_reward[done_index] = 0

        return self._should_continue(self.num_timesteps)

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
            self._record_completed_episode(episode_rewards[done_index], terminal_infos[done_index])

        self.episode_reward = np.zeros_like(episode_rewards)
        return self._should_continue(context.end_timestep)

    def _record_completed_episode(self, reward_value: float, info: dict) -> None:
        """Append one finished episode's total reward to the pruning window, unless discarded."""
        if not info.get(constants.DISCARD_INFO_KEY, False):
            self.episode_rewards.append(reward_value)

    def _should_continue(self, current_timestep: int) -> bool:
        """Prune once the reward window is full and its mean falls below the threshold."""
        reward_window_is_full = len(self.episode_rewards) == self.episode_rewards.maxlen
        if current_timestep <= self.pruning_start_at or not reward_window_is_full:
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
            self._handle_first_step(agent_index, reset_info)

        # Write reset info at each reset (when done=True, reset has already happened and reset_info is available)
        for done_index in np.where(self.locals["dones"] == True)[0]:
            self._handle_episode_end(done_index, self.locals["reset_infos"][done_index])

        return True

    def process_episode(self, context: EpisodeCallbackContext) -> bool:
        """Log initial and post-terminal reset information once per episode.

        Reset info only ever appears at an episode's first transition (the initial
        reset) or its terminal one (the reset that followed `done`), so - like
        `_handle_first_step`/`_handle_episode_end` below - only those two rows of
        the batch need inspecting, not every transition in between.
        """
        batch = context.batch
        agent_index = self._get_agent_index(context)
        for reset_info in get_episode_reset_infos(batch, 0):
            self._handle_first_step(agent_index, reset_info)

        terminal_index = batch.transition_count - 1
        terminal_dones = slice_episode_field(batch, "dones", terminal_index)
        terminal_reset_infos = get_episode_reset_infos(batch, terminal_index)
        for done_index in np.flatnonzero(terminal_dones):
            self._handle_episode_end(agent_index, terminal_reset_infos[done_index])

        return True

    def _handle_first_step(self, agent_index: int, reset_info: dict) -> None:
        if agent_index in self.first_step_tracker:
            return
        self.episode_counter[agent_index] = 0
        self.first_step_tracker.append(agent_index)
        self._log_reset_info(agent_index, reset_info)

    def _handle_episode_end(self, agent_index: int, reset_info: dict) -> None:
        self.episode_counter[agent_index] = self.episode_counter.get(agent_index, 0) + 1
        self._log_reset_info(agent_index, reset_info)

    def _log_reset_info(self, agent_index: int, reset_info: dict) -> None:
        if not reset_info:
            return
        self.connector.log_dict(
            reset_info,
            f"Reset Info - Agent {agent_index} - Episode {self.episode_counter[agent_index]}",
        )


class GammaScheduleCallback(EpisodeBatchableCallbackMixin, BaseCallback):
    """
    Sets the discount factor (gamma) from a schedule before every rollout.

    The schedule follows the SB3 learning-rate convention: it is called with `progress_remaining`, which goes from 1
    (start of training) to 0 (end of training).
    """

    def __init__(self, gamma_schedule: Callable[[float], float], verbose=0):
        """
        Args:
            gamma_schedule: Callable mapping `progress_remaining` to the gamma value to use for the next rollout.
            verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
        """
        super().__init__(verbose)
        self.gamma_schedule = gamma_schedule

    def _on_rollout_start(self) -> None:
        # Computed here instead of reading `model._current_progress_remaining`, which SB3 only updates after a rollout
        # (and which still holds the previous run's value on the first rollout of a loaded model).
        progress_remaining = 1.0 - float(self.model.num_timesteps) / float(self.model._total_timesteps)
        gamma = float(self.gamma_schedule(progress_remaining))

        self.model.gamma = gamma
        # Buffers copy gamma on creation (on-policy GAE, n-step replay), so they need to be updated as well.
        for buffer_name in ("rollout_buffer", "replay_buffer"):
            buffer = getattr(self.model, buffer_name, None)
            if buffer is not None and hasattr(buffer, "gamma"):
                buffer.gamma = gamma

        self.logger.record("train/gamma", gamma)

    def _on_step(self) -> bool:
        return True

    def process_episode(self, context: EpisodeCallbackContext) -> bool:
        return True

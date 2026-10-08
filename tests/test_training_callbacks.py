"""SB3 training callbacks driven directly: per-step (`_on_step` with SB3 locals) and per-episode (`process_episode`).

Strict xfails document unresolved findings; they assert the desired behavior.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from async_gym_agents.data_classes import (
    EpisodeBatch,
    EpisodeCallbackContext,
    EpisodeKind,
)
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

from rl_framework.util import (
    GammaScheduleCallback,
    LoggingCallback,
    ResetInfoCallback,
    SavingCallback,
    add_callbacks_to_callback,
)
from rl_framework.util.sb3_training_callbacks import ExperimentPruningCallback
from tests.toys import RecordingConnector


def set_step(callback, rewards, dones, infos=None, actions=None, reset_infos=None, num_timesteps=1):
    n = len(rewards)
    callback.locals = {
        "new_obs": np.zeros((n, 1)),
        "actions": np.array(actions if actions is not None else [0] * n),
        "rewards": np.array(rewards, dtype=np.float32),
        "dones": np.array(dones),
        "infos": infos or [{} for _ in range(n)],
    }
    if reset_infos is not None:
        callback.locals["reset_infos"] = reset_infos
    callback.num_timesteps = num_timesteps
    return callback._on_step()


def episode(rewards, infos=None, reset_infos=None, start_timestep=0):
    """One single-env on-policy episode whose last transition is done."""
    n = len(rewards)
    batch = EpisodeBatch(
        episode_kind=EpisodeKind.ON_POLICY,
        transition_count=n,
        fields={
            "new_obs": np.zeros((n, 1), np.float32),
            "actions": np.zeros(n, np.int64),
            "environment_rewards": np.array(rewards, np.float32),
            "dones": np.array([False] * (n - 1) + [True]),
        },
        infos=infos or {},
        reset_infos=reset_infos or {},
    )
    return EpisodeCallbackContext(batch=batch, start_timestep=start_timestep, end_timestep=start_timestep + n)


class Noop(BaseCallback):
    def _on_step(self):
        return True


def scalars(connector, name="Episode reward"):
    return connector.value_sequences_to_log.get(name, [])


# --- add_callbacks_to_callback ---


def test_add_callbacks_appends_missing_callbacks_to_a_callback_list_without_modifying_it():
    first, second = Noop(), Noop()
    target = CallbackList([first])

    result = add_callbacks_to_callback(CallbackList([first, second]), target)

    assert result.callbacks == [first, second]
    assert target.callbacks == [first]


@pytest.mark.parametrize("make_target", [lambda single: single, lambda single: [single]], ids=["single", "list"])
def test_add_callbacks_to_a_single_callback_makes_them_reachable(make_target):
    single = Noop()
    added = Noop()

    result = add_callbacks_to_callback(CallbackList([added]), make_target(single))

    assert isinstance(result, CallbackList)
    assert result.callbacks == [single, added]


def test_add_callbacks_to_no_callback():
    added = Noop()
    assert add_callbacks_to_callback(CallbackList([added]), None).callbacks == [added]


# --- LoggingCallback ---


def test_logging_callback_logs_episode_reward_of_each_done_env():
    connector = RecordingConnector()
    callback = LoggingCallback(connector)
    set_step(callback, [1, 2], [False, False], num_timesteps=1)
    set_step(callback, [1, 2], [True, False], num_timesteps=2)
    set_step(callback, [1, 2], [False, True], num_timesteps=3)

    assert scalars(connector) == [(2, 2.0), (3, 6.0)]


def test_logging_callback_logs_every_logging_frequency_episodes_the_window_mean():
    connector = RecordingConnector()
    callback = LoggingCallback(connector, logging_frequency=2)
    for timestep, reward in enumerate([1, 3, 5, 7], start=1):
        set_step(callback, [reward], [True], num_timesteps=timestep)

    assert scalars(connector) == [(2, 2.0), (4, 6.0)]


def test_logging_callback_logs_meta_infos_once_until_they_change():
    connector = RecordingConnector()
    callback = LoggingCallback(connector)
    set_step(callback, [0], [False], infos=[{"meta_map": "desert", "meta_cfg": {"a": 1}}])
    set_step(callback, [0], [False], infos=[{"meta_map": "desert", "meta_cfg": {"a": 1}}])
    assert connector.dicts_to_log == {"meta_map": {"meta_map": "desert"}, "meta_cfg": {"a": 1}}

    connector.dicts_to_log.clear()
    set_step(callback, [0], [False], infos=[{"meta_map": "forest", "meta_cfg": {"a": 1}}])
    assert connector.dicts_to_log == {"meta_map": {"meta_map": "forest"}}


@pytest.mark.parametrize(
    "previous, current, expected",
    [
        (1, 1, True),
        (1, 2, False),
        (np.array([1, 2]), np.array([1, 2]), True),
        (np.array([1, 2]), np.array([1, 3]), False),
        (np.array([1, 2]), np.array([1, 2, 3]), False),
    ],
)
def test_metadata_comparison_handles_arrays(previous, current, expected):
    assert LoggingCallback._metadata_matches(previous, current) is expected


def test_logging_callback_logs_action_distributions_when_requested():
    connector = RecordingConnector()
    callback = LoggingCallback(connector, log_distributions=True)
    set_step(callback, [0], [False], actions=[1])
    set_step(callback, [0], [True], actions=[0], num_timesteps=2)

    assert [
        (values.tolist(), step) for values, step in connector.histogram_sequences_to_log["Action distribution"]
    ] == [([1, 0], 2)]


def test_logging_callback_survives_a_discarded_first_episode():
    connector = RecordingConnector()
    callback = LoggingCallback(connector)
    set_step(callback, [1], [True], infos=[{"discard": True}])
    set_step(callback, [2], [True], num_timesteps=2)

    assert scalars(connector) == [(2, 2.0)]


def test_logging_callback_logs_no_nan_reward_for_a_discarded_episode():
    connector = RecordingConnector()
    callback = LoggingCallback(connector)
    set_step(callback, [1], [True], num_timesteps=1)
    set_step(callback, [2], [True], infos=[{"discard": True}], num_timesteps=2)

    assert scalars(connector) == [(1, 1.0)]


def test_discarded_episodes_do_not_count_towards_the_logging_frequency():
    connector = RecordingConnector()
    callback = LoggingCallback(connector, logging_frequency=2)
    set_step(callback, [1], [True], num_timesteps=1)
    set_step(callback, [5], [True], infos=[{"discard": True}], num_timesteps=2)
    set_step(callback, [3], [True], num_timesteps=3)

    assert scalars(connector) == [(3, 2.0)]


def test_logging_callback_process_episode_skips_discarded_episodes():
    connector = RecordingConnector()
    callback = LoggingCallback(connector)

    callback.process_episode(episode([1, 2], infos={1: [{"discard": True}]}))
    callback.process_episode(episode([4], start_timestep=2))

    assert scalars(connector) == [(3, 4.0)]


def test_logging_callback_process_episode_matches_step_wise_logging():
    connector = RecordingConnector()
    callback = LoggingCallback(connector)
    infos = {0: [{"step_metric_x": 1.0}], 2: [{"step_metric_x": 3.0, "meta_level": 4}]}

    assert callback.process_episode(episode([1, 2, 3], infos=infos, start_timestep=10)) is True

    assert scalars(connector) == [(13, 6.0)]
    assert scalars(connector, "Mean - x") == [(13, 2.0)]
    assert connector.dicts_to_log == {"meta_level": {"meta_level": 4}}


def test_episode_batchable_mixin_advances_counters():
    callback = LoggingCallback(RecordingConnector())
    callback.advance_callback(transition_count=5, num_timesteps=17)
    callback.advance_callback(transition_count=2, num_timesteps=19)

    assert (callback.n_calls, callback.num_timesteps) == (7, 19)


# --- SavingCallback ---


def test_saving_callback_uploads_after_each_checkpoint_frequency_steps():
    connector = RecordingConnector()
    agent = object()
    callback = SavingCallback(agent, connector, checkpoint_frequency=3)
    for timestep in range(1, 10):
        callback.num_timesteps = timestep
        assert callback._on_step() is True

    assert [upload["checkpoint_id"] for upload in connector.uploads] == [4, 8]
    assert all(upload["agent"] is agent for upload in connector.uploads)


def test_saving_callback_process_episode_does_not_upload_before_first_checkpoint():
    connector = RecordingConnector()
    callback = SavingCallback(object(), connector, checkpoint_frequency=100)
    callback.process_episode(episode([0] * 10, start_timestep=0))

    assert connector.uploads == []


# --- ExperimentPruningCallback ---


def fill_pruning_window(callback, reward, num_timesteps):
    result = True
    for _ in range(callback.episode_rewards.maxlen):
        result = set_step(callback, [reward], [True], num_timesteps=num_timesteps)
    return result


def test_pruning_stops_training_when_mean_reward_is_below_threshold():
    callback = ExperimentPruningCallback(episode_reward_threshold=1.0, pruning_start_at=10)
    assert fill_pruning_window(callback, reward=0.5, num_timesteps=11) is False


def test_pruning_continues_when_mean_reward_reaches_threshold():
    callback = ExperimentPruningCallback(episode_reward_threshold=1.0, pruning_start_at=10)
    assert fill_pruning_window(callback, reward=1.0, num_timesteps=11) is True


def test_pruning_waits_for_pruning_start_and_full_window():
    callback = ExperimentPruningCallback(episode_reward_threshold=1.0, pruning_start_at=10)
    assert fill_pruning_window(callback, reward=0.0, num_timesteps=10) is True

    callback = ExperimentPruningCallback(episode_reward_threshold=1.0, pruning_start_at=10)
    assert set_step(callback, [0.0], [True], num_timesteps=1000) is True


def test_pruning_ignores_discarded_episodes_and_sums_episode_rewards():
    callback = ExperimentPruningCallback()
    set_step(callback, [1, 1], [False, False])
    set_step(callback, [2, 2], [True, True], infos=[{"discard": True}, {}])

    assert list(callback.episode_rewards) == [3.0]
    np.testing.assert_array_equal(callback.episode_reward, [0, 0])


def test_pruning_process_episode_records_the_episode_reward():
    callback = ExperimentPruningCallback(episode_reward_threshold=10.0, pruning_start_at=0)

    assert callback.process_episode(episode([1, 2, 3])) is True
    assert list(callback.episode_rewards) == [6.0]


# --- ResetInfoCallback ---


def test_reset_info_callback_logs_first_reset_and_resets_after_done():
    connector = RecordingConnector()
    callback = ResetInfoCallback(connector)
    set_step(callback, [0, 0], [False, False], reset_infos=[{"seed": 1}, {}])
    set_step(callback, [0, 0], [False, False], reset_infos=[{"seed": 1}, {}])
    set_step(callback, [0, 0], [True, False], reset_infos=[{"seed": 2}, {}])

    assert connector.dicts_to_log == {
        "Reset Info - Agent 0 - Episode 0": {"seed": 1},
        "Reset Info - Agent 0 - Episode 1": {"seed": 2},
    }


def test_reset_info_callback_falls_back_to_training_env_reset_infos():
    connector = RecordingConnector()
    callback = ResetInfoCallback(connector)
    callback.model = SimpleNamespace(get_env=lambda: SimpleNamespace(reset_infos=[{"seed": 9}]))
    set_step(callback, [0], [False])

    assert connector.dicts_to_log == {"Reset Info - Agent 0 - Episode 0": {"seed": 9}}


def test_reset_info_callback_process_episode_logs_initial_and_terminal_reset_infos():
    connector = RecordingConnector()
    callback = ResetInfoCallback(connector)
    context = episode([0, 0, 0], reset_infos={0: [{"seed": 1}], 2: [{"seed": 2}]})

    assert callback.process_episode(context) is True
    assert connector.dicts_to_log == {
        "Reset Info - Agent 0 - Episode 0": {"seed": 1},
        "Reset Info - Agent 0 - Episode 1": {"seed": 2},
    }


# --- GammaScheduleCallback ---


def test_gamma_schedule_callback_sets_gamma_on_model_and_buffers():
    recorded = {}
    callback = GammaScheduleCallback(lambda progress_remaining: 0.9 + 0.1 * (1 - progress_remaining))
    callback.model = SimpleNamespace(
        num_timesteps=25,
        _total_timesteps=100,
        gamma=0.0,
        rollout_buffer=SimpleNamespace(gamma=0.0),
        replay_buffer=SimpleNamespace(),
        logger=SimpleNamespace(record=lambda key, value: recorded.update({key: value})),
    )

    callback._on_rollout_start()

    assert callback.model.gamma == pytest.approx(0.925)
    assert callback.model.rollout_buffer.gamma == pytest.approx(0.925)
    assert not hasattr(callback.model.replay_buffer, "gamma")
    assert recorded == {"train/gamma": pytest.approx(0.925)}
    assert callback._on_step() is True
    assert callback.process_episode(episode([0])) is True

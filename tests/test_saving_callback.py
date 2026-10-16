"""SavingCallback checkpoint uploads in async episode-batched mode."""

from unittest.mock import Mock

from async_gym_agents.data_classes import EpisodeCallbackContext

from rl_framework.util import SavingCallback


def uploads_for_episode(start_timestep, end_timestep, checkpoint_frequency=2):
    connector = Mock()
    callback = SavingCallback(agent=Mock(), connector=connector, checkpoint_frequency=checkpoint_frequency)
    callback.process_episode(
        EpisodeCallbackContext(
            batch=None, start_timestep=start_timestep, end_timestep=end_timestep, worker_index=0, env_index=0
        )
    )
    return [call.kwargs["checkpoint_id"] for call in connector.upload.call_args_list]


def test_episode_crossing_one_checkpoint_uploads_once():
    assert uploads_for_episode(0, 3) == [3]


def test_episode_crossing_several_checkpoints_uploads_once():
    assert uploads_for_episode(0, 10) == [3]


def test_checkpoint_schedule_continues_after_a_long_episode():
    connector = Mock()
    callback = SavingCallback(agent=Mock(), connector=connector, checkpoint_frequency=2)
    for start, end in [(0, 10), (10, 11), (11, 12)]:
        callback.process_episode(
            EpisodeCallbackContext(batch=None, start_timestep=start, end_timestep=end, worker_index=0, env_index=0)
        )

    # Upload at 3 (first boundary of the long episode), skipped boundaries 5, 7, 9; next boundary 11 is crossed at 12
    assert [call.kwargs["checkpoint_id"] for call in connector.upload.call_args_list] == [3, 12]

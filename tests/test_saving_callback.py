"""SavingCallback checkpoint uploads in async episode-batched mode.

Strict xfails document unresolved findings; they assert the desired behavior.
"""

from unittest.mock import Mock

import pytest
from async_gym_agents.data_classes import EpisodeCallbackContext

from rl_framework.util import SavingCallback


def uploads_for_episode(start_timestep, end_timestep, checkpoint_frequency=2):
    connector = Mock()
    callback = SavingCallback(agent=Mock(), connector=connector, checkpoint_frequency=checkpoint_frequency)
    callback.process_episode(
        EpisodeCallbackContext(batch=None, start_timestep=start_timestep, end_timestep=end_timestep)
    )
    return [call.kwargs["checkpoint_id"] for call in connector.upload.call_args_list]


def test_episode_crossing_one_checkpoint_uploads_once():
    assert uploads_for_episode(0, 3) == [3]


@pytest.mark.xfail(
    strict=True,
    reason="sb3_training_callbacks.py:231-236 replays every crossed boundary, uploading the unchanged model each time",
)
def test_episode_crossing_several_checkpoints_uploads_once():
    assert len(uploads_for_episode(0, 10)) == 1

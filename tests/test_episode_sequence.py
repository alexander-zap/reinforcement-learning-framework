"""EpisodeSequence: construction from episodes/generators/datasets, saving, and conversion to other formats."""

from pathlib import Path

import numpy as np
import pytest
from imitation.data.types import TrajectoryWithRew

from rl_framework.agent.imitation import EpisodeSequence
from rl_framework.agent.imitation.episode_sequence import WrappedEpisodeSequence

CARTPOLE_ROLLOUT = Path(__file__).parents[1] / "data" / "cartpole_rollout"


def trajectory(length=3, terminal=True, offset=0):
    return TrajectoryWithRew(
        obs=np.arange((length + 1) * 2, dtype=np.float32).reshape(length + 1, 2) + offset,
        acts=np.arange(length, dtype=np.int64),
        infos=None,
        terminal=terminal,
        rews=np.arange(length, dtype=np.float32) + 1,
    )


def assert_same_trajectory(actual, expected):
    np.testing.assert_array_equal(actual.obs, expected.obs)
    np.testing.assert_array_equal(actual.acts, expected.acts)
    np.testing.assert_array_equal(actual.rews, expected.rews)
    assert actual.terminal == expected.terminal


@pytest.fixture
def episodes():
    return [trajectory(3, terminal=True), trajectory(2, terminal=False, offset=100)]


def test_empty_sequence():
    sequence = EpisodeSequence()
    assert len(sequence) == 0
    assert not sequence


def test_from_episodes_keeps_every_episode(episodes):
    sequence = EpisodeSequence.from_episodes(episodes)

    assert len(sequence) == 2
    for actual, expected in zip(sequence, episodes):
        assert_same_trajectory(actual, expected)


def test_from_episode_generator_takes_the_requested_number_of_episodes(episodes):
    def generate():
        while True:
            yield from episodes

    sequence = EpisodeSequence.from_episode_generator(generate(), n_episodes=3)

    assert len(sequence) == 3
    assert_same_trajectory(sequence[2], episodes[0])


def test_save_and_load_round_trip(tmp_path, episodes):
    EpisodeSequence.from_episodes(episodes).save(tmp_path / "episodes")

    loaded = EpisodeSequence.from_dataset(str(tmp_path / "episodes"))

    assert len(loaded) == 2
    assert_same_trajectory(loaded[1], episodes[1])


def test_recorded_cartpole_rollout_can_be_loaded():
    sequence = EpisodeSequence.from_dataset(str(CARTPOLE_ROLLOUT))

    assert len(sequence) == 20
    assert sequence[0].obs.shape[1] == 4
    assert len(sequence[0].obs) == len(sequence[0].acts) + 1


def test_imitation_episodes_are_the_sequence_itself(episodes):
    sequence = EpisodeSequence.from_episodes(episodes)
    assert sequence.to_imitation_episodes() is sequence


def test_generic_episodes_are_transition_tuples(episodes):
    generic = EpisodeSequence.from_episodes(episodes).to_generic_episodes()

    assert isinstance(generic, WrappedEpisodeSequence)
    first = generic[0]
    assert len(first) == 3
    obs, action, next_obs, reward, terminated, truncated, info = first[0]
    np.testing.assert_array_equal(obs, [0, 1])
    np.testing.assert_array_equal(next_obs, [2, 3])
    assert (action, reward, terminated, truncated, info) == (0, 1.0, False, False, {})
    assert first[-1][4:6] == (True, False)

    second = generic[1]
    assert len(second) == 2
    assert second[-1][4:6] == (False, True)


def test_d3rlpy_episodes_drop_the_final_observation_and_add_action_and_reward_axes(episodes):
    d3rlpy_episode = EpisodeSequence.from_episodes(episodes).to_d3rlpy_episodes()[0]

    assert d3rlpy_episode.observations.shape == (3, 2)
    assert d3rlpy_episode.actions.shape == (3, 1)
    assert d3rlpy_episode.rewards.shape == (3, 1)
    assert d3rlpy_episode.terminated


def test_additional_conversions_are_chained(episodes):
    generic = EpisodeSequence.from_episodes(episodes).to_generic_episodes()

    lengths = generic.episode_sequence_from_additional_conversion(len)

    assert [lengths[0], lengths[1]] == [3, 2]
    assert len(lengths) == 2

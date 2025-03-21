import itertools
from typing import (
    Generator,
    Iterable,
    List,
    Optional,
    Sequence,
    Sized,
    Tuple,
    Union,
    cast,
)

import d3rlpy
import imitation
import imitation.data.types
import numpy as np
from imitation.data import serialize

from rl_framework.util import SizedGenerator, patch_datasets

patch_datasets()

GenericEpisode = List[Tuple[object, object, object, float, bool, bool, dict]]


class EpisodeSequence(Iterable[GenericEpisode], Sized):
    """
    Class to load, transform and iterate over episodes, optimized for memory efficiency.
        - Using HuggingFace "load_from_disk" for loading
        - Using generators for underlying data management
        - Format changing transformations also return generators

    The sampling logic and order of episode generation is fully controlled by the `self._episode_generator` attribute.

    Each episode consists of a sequence, which has the following format:
        [
            (obs_t0, action_t0, next_obs_t0, reward_t0, terminated_t0, truncated_t0, info_t0),
            (obs_t1, action_t1, next_obs_t1, reward_t1, terminated_t1, truncated_t1, info_t1),
            ...
        ]
        Interpretation: Transition from obs to next_obs with action, receiving reward.
            Additional information returned about transition to next_obs: terminated, truncated and info.

    NOTE: Please remember that any results of the format changing `to_` operation will be using the same generator.
        Iterating over the resulting format-changed generators still consumes the internal generator of EpisodeSequence.
    """

    def __init__(self):
        def empty() -> Generator:
            yield from ()

        self._episode_generator: Generator[GenericEpisode, None, None] = empty()
        self._len = 0
        self._looping = False

    def __len__(self):
        return self._len

    def __iter__(self):
        return self._episode_generator

    @staticmethod
    def from_episode_generator(
        episode_generator: Generator[GenericEpisode, None, None], n_episodes: int
    ) -> "EpisodeSequence":
        """
        Initialize an EpisodeSequence based on a provided generator (of GenericEpisode objects).

        Args:
            episode_generator (Generator): Custom episode generator generating GenericEpisodes
                every time __next__() is called.
            n_episodes (int): Amount of episodes the generator will generate (to limit infinite generators).
                If amount of generated elements from generator are already known, pass it as "n_episodes".

        NOTE: Loading an EpisodeSequence from a generator does not support looping.
            If this behavior is wished, make sure the `episode_generator` itself is looping instead.

        NOTE: Loading an EpisodeSequence from a generator does not support splitting.
            If a split into multiple EpisodeSequences is wished, create two generators and load from both individually.

        Returns:
            episode_sequence: Representation of episode sequence (this class).
        """
        episode_sequence = EpisodeSequence()
        episode_sequence._episode_generator = itertools.islice(episode_generator, n_episodes)
        episode_sequence._len = n_episodes
        episode_sequence._looping = False
        return episode_sequence

    @staticmethod
    def from_episodes(
        episodes: Sequence[GenericEpisode], loop: bool = False, split_by_fractions: Optional[List[float]] = None
    ) -> Union["EpisodeSequence", List["EpisodeSequence"]]:
        """
        Initialize an EpisodeSequence based on a sequence of GenericEpisode objects.

        Args:
            episodes (Sequence[GenericEpisode]): Episodes in generic format.
            loop (bool): Flag whether the generator should loop over the episodes repeatedly.
            split_by_fractions (Optional): List of fraction positions by which the dataset should be split.
                If provided, this method returns len(split_by_fractions) + 1 unique EpisodeSequences.
                All elements of this list should be in ]0.0, 1.0[.

        Returns:
            episode_sequence: Representation of episode sequence (this class).
                Returns multiple episode sequences (in a list) if `split_by_fractions` parameter is given.
        """

        def generate_episodes(
            owner_of_generator: EpisodeSequence, generic_episodes: Sequence[GenericEpisode], indices: List[int]
        ) -> Generator[GenericEpisode, None, None]:
            while True:
                for index in indices:
                    yield generic_episodes[index]
                if not owner_of_generator._looping:
                    break

        def create_episode_sequence(indices: List[int]) -> EpisodeSequence:
            episode_sequence = EpisodeSequence()
            episode_sequence._episode_generator = generate_episodes(episode_sequence, episodes, indices)
            episode_sequence._len = len(episodes)
            episode_sequence._looping = loop
            return episode_sequence

        if split_by_fractions:
            n_episodes = len(episodes)
            assert all(isinstance(fraction, float) for fraction in split_by_fractions)
            assert all(0.0 < fraction < 1.0 for fraction in split_by_fractions)
            split_by_indices = [int(fraction * n_episodes) for fraction in split_by_fractions]
            split_by_indices = sorted(split_by_indices)

            episode_sequences = []
            for start_index, end_index in zip([0] + split_by_indices, split_by_indices + [n_episodes]):
                split_indices = list(range(start_index, end_index))
                episode_sequence = create_episode_sequence(split_indices)
                episode_sequences.append(episode_sequence)
            return episode_sequences
        else:
            all_indices = list(range(len(episodes)))
            episode_sequence = create_episode_sequence(all_indices)
            return episode_sequence

    @staticmethod
    def from_dataset(
        file_path: str, loop: bool = False, split_by_fractions: Optional[List[float]] = None
    ) -> Union["EpisodeSequence", List["EpisodeSequence"]]:
        """
        Initialize an EpisodeSequence based on provided huggingface dataset path.

        Episode sequences are loaded from a provided file path in the agent section of the config.
        Files of recorded episode sequences are generated by saving Trajectory objects (`imitation` library).
        https://imitation.readthedocs.io/en/latest/main-concepts/trajectories.html#storing-loading-trajectories

        Args:
            file_path (str): Path to huggingface dataset recording of episodes.
            loop (bool): Flag whether the generator should loop over the dataset repeatedly.s
            split_by_fractions (Optional): List of fraction positions by which the dataset should be split.
                If provided, this method returns len(split_by_fractions) + 1 unique EpisodeSequences.
                All elements of this list should be in ]0.0, 1.0[.

        Returns:
            episode_sequence: Representation of episode sequence (this class).
                Returns multiple episode sequences (in a list) if `split_by_fractions` parameter is given.
        """

        def generate_episodes(
            owner_of_generator: EpisodeSequence,
            imitation_trajectories: Sequence[imitation.data.types.TrajectoryWithRew],
            indices: List[int],
        ) -> Generator[GenericEpisode, None, None]:
            while True:
                for index in indices:
                    trajectory = imitation_trajectories[index]
                    obs = trajectory.obs[:-1]
                    acts = trajectory.acts
                    rews = trajectory.rews
                    next_obs = trajectory.obs[1:]
                    terminations = np.zeros(len(trajectory.acts), dtype=bool)
                    truncations = np.zeros(len(trajectory.acts), dtype=bool)
                    terminations[-1] = trajectory.terminal
                    truncations[-1] = not trajectory.terminal
                    infos = np.array([{}] * len(trajectory)) if trajectory.infos is None else trajectory.infos
                    episode: GenericEpisode = list(zip(*[obs, acts, next_obs, rews, terminations, truncations, infos]))
                    yield episode
                if not owner_of_generator._looping:
                    break

        def create_episode_sequence(indices: List[int]) -> EpisodeSequence:
            episode_sequence = EpisodeSequence()
            episode_sequence._episode_generator = generate_episodes(episode_sequence, trajectories, indices)
            episode_sequence._len = len(indices)
            episode_sequence._looping = loop
            return episode_sequence

        trajectories = cast(Sequence[imitation.data.types.TrajectoryWithRew], serialize.load(file_path))

        if split_by_fractions:
            n_trajectories = len(trajectories)
            assert all(isinstance(fraction, float) for fraction in split_by_fractions)
            assert all(0.0 < fraction < 1.0 for fraction in split_by_fractions)
            split_by_indices = [int(fraction * n_trajectories) for fraction in split_by_fractions]
            split_by_indices = sorted(split_by_indices)

            episode_sequences = []
            for start_index, end_index in zip([0] + split_by_indices, split_by_indices + [n_trajectories]):
                split_indices = list(range(start_index, end_index))
                episode_sequence = create_episode_sequence(split_indices)
                episode_sequences.append(episode_sequence)
            return episode_sequences
        else:
            all_indices = list(range(len(trajectories)))
            episode_sequence = create_episode_sequence(all_indices)
            return episode_sequence

    def save(self, file_path):
        """
        Save episode sequence into a file, saved as HuggingFace dataset.

        Args:
            file_path: File path and file name to save episode sequence to.
        """
        self._looping = False
        trajectories: Sequence[imitation.data.types.TrajectoryWithRew] = list(self.to_imitation_episodes())
        serialize.save(file_path, trajectories)

    def to_imitation_episodes(self) -> SizedGenerator[imitation.data.types.TrajectoryWithRew]:
        def generate_imitation_episodes():
            for generic_episode in self._episode_generator:
                observations, actions, next_observations, rewards, terminations, truncations, infos = (
                    np.array(x) for x in list(zip(*generic_episode))
                )
                observations = np.expand_dims(observations, axis=1) if observations.ndim == 1 else observations
                next_observations = (
                    np.expand_dims(next_observations, axis=1) if next_observations.ndim == 1 else next_observations
                )
                all_observations = np.vstack([observations, next_observations[-1:]])
                episode_trajectory = imitation.data.types.TrajectoryWithRew(
                    obs=all_observations, acts=actions, rews=rewards, infos=infos, terminal=terminations[-1]
                )
                yield episode_trajectory

        return SizedGenerator(generate_imitation_episodes(), len(self), self._looping)

    def to_d3rlpy_episodes(self) -> SizedGenerator[d3rlpy.dataset.components.Episode]:
        def generate_d3rlpy_episodes():
            for generic_episode in self._episode_generator:
                observations, actions, next_observations, rewards, terminations, truncations, infos = (
                    np.array(x) for x in list(zip(*generic_episode))
                )
                rewards = np.expand_dims(rewards, axis=1)
                actions = np.expand_dims(actions, axis=1) if actions.ndim == 1 else actions
                episode = d3rlpy.dataset.components.Episode(
                    observations=observations,
                    actions=actions,
                    rewards=rewards,
                    terminated=terminations[-1],
                )
                yield episode

        return SizedGenerator(generate_d3rlpy_episodes(), len(self), self._looping)

    def to_generic_episodes(self) -> SizedGenerator[GenericEpisode]:
        def generate_generic_episodes():
            return self._episode_generator

        return SizedGenerator(generate_generic_episodes(), len(self), self._looping)

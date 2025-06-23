import uuid
from typing import Generator, Generic, List, Sequence, Tuple, TypeVar, cast

import d3rlpy
import datasets
import imitation
import imitation.data.types
import numpy as np
from imitation.data import serialize
from imitation.data.huggingface_utils import (
    TrajectoryDatasetSequence,
    trajectories_to_dataset,
    trajectories_to_dict,
)

from rl_framework.util import SizedGenerator, patch_datasets

patch_datasets()

GenericEpisode = List[Tuple[object, object, object, float, bool, bool, dict]]

T = TypeVar("T")
IN = TypeVar("IN")
OUT = TypeVar("OUT")

"""

to_sequence(converter: Converter[T]) -> EpisodeSequence[T]

ReadConverter[T] (z.b. from_dataset wäre das TrajectoryWithReward)
WriteConverter[T] (z.b.: db3ddingsi)

If ReadConverter == WriteConverter: skip
else: WriteConverter(ReadConverter)


"""

# CONVERTER_REGISTRY = {
#     "GENERIC": GenericEpisodeConverter,
#     "IMITATION": ImitationConverter,
#     "D3RLPY": D3RLPYConverter
# }
#
#
# class Converter(Generic[IN, OUT]):
#     def __init__(self, converter: str):
#         self.converter = CONVERTER_REGISTRY[converter]()
#
#     def __call__(self, episode: IN) -> OUT:
#         return self.converter(episode)


def generate_generic_episode_from_imitation_trajectory(
    trajectory: imitation.data.types.TrajectoryWithRew,
) -> GenericEpisode:
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
    return episode


def generate_imitation_trajectory_from_generic_episode(
    generic_episode: GenericEpisode,
) -> imitation.data.types.TrajectoryWithRew:
    observations, actions, next_observations, rewards, terminations, truncations, infos = (
        np.array(x) for x in list(zip(*generic_episode))
    )
    observations = np.expand_dims(observations, axis=1) if observations.ndim == 1 else observations
    next_observations = np.expand_dims(next_observations, axis=1) if next_observations.ndim == 1 else next_observations
    all_observations = np.vstack([observations, next_observations[-1:]])
    episode_trajectory = imitation.data.types.TrajectoryWithRew(
        obs=all_observations, acts=actions, rews=rewards, infos=infos, terminal=terminations[-1]
    )
    return episode_trajectory


def generate_d3rlpy_episode_from_generic_episode(generic_episode: GenericEpisode) -> d3rlpy.dataset.components.Episode:
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
    return episode


class EpisodeSequence(Sequence[T], Generic[T]):
    """
    Class to load, transform and access episodes, optimized for memory efficiency.
        - Using HuggingFace datasets for memory-efficient data management (using arrow datasets under the hood)
        - Using converters for format changing transformations

    Each episode consists of a sequence, which has the following format:
        [
            (obs_t0, action_t0, next_obs_t0, reward_t0, terminated_t0, truncated_t0, info_t0),
            (obs_t1, action_t1, next_obs_t1, reward_t1, terminated_t1, truncated_t1, info_t1),
            ...
        ]
        Interpretation: Transition from obs to next_obs with action, receiving reward.
            Additional information returned about transition to next_obs: terminated, truncated and info.

    """

    def __init__(self):
        self._episodes: Sequence[imitation.data.types.TrajectoryWithRew] = []

    def __len__(self):
        return len(self._episodes)

    def __getitem__(self, index) -> T:
        raise self._episodes[index]

    @staticmethod
    def from_episode_generator(
        # FIXME: Type should be T
        episode_generator: Generator[imitation.data.types.TrajectoryWithRew, None, None],
        n_episodes: int,
    ) -> "EpisodeSequence":
        """
        Initialize an EpisodeSequence based on a provided episode generator.

        Args:
            episode_generator (Generator): Custom episode generator generating episodes of type T.
            n_episodes (int): Amount of episodes the generator will generate (to limit infinite generators).

        Returns:
            episode_sequence: Representation of episode sequence (this class).
        """

        # NOTE: This is a hack to make the generator pickleable (because of huggingface datasets caching requirements)
        #  https://github.com/huggingface/datasets/issues/6194#issuecomment-1708080653
        class TrajectoryGenerator:
            def __init__(self, generator, trajectories_to_generate):
                self.generator_id = str(uuid.uuid4())
                self.generator = generator
                self.trajectories_to_generate = trajectories_to_generate

            def __call__(self, *args, **kwargs):
                for _ in range(self.trajectories_to_generate):
                    imitation_trajectory = next(self.generator)
                    trajectory_dict = {
                        key: value[0] for key, value in trajectories_to_dict([imitation_trajectory]).items()
                    }
                    yield trajectory_dict

            def __reduce__(self):
                return raise_pickling_error, (self.generator_id,)

        def raise_pickling_error(*args, **kwargs):
            raise AssertionError("Cannot actually pickle TrajectoryGenerator!")

        episode_sequence = EpisodeSequence()
        trajectory_dataset = datasets.Dataset.from_generator(TrajectoryGenerator(episode_generator, n_episodes))
        trajectory_dataset_sequence = TrajectoryDatasetSequence(trajectory_dataset)
        episode_sequence._episodes = cast(Sequence[imitation.data.types.TrajectoryWithRew], trajectory_dataset_sequence)
        return episode_sequence

    @staticmethod
    def from_episodes(episodes: Sequence[imitation.data.types.TrajectoryWithRew]) -> "EpisodeSequence":
        """
        Initialize an EpisodeSequence based on a sequence of episode objects.

        Args:
            episodes (Sequence): Sequence of episodes of type T.

        Returns:
            episode_sequence: Representation of episode sequence (this class).
        """

        episode_sequence = EpisodeSequence()
        trajectory_dataset = trajectories_to_dataset(episodes)
        trajectories_dataset_sequence = TrajectoryDatasetSequence(trajectory_dataset)
        episode_sequence._episodes = cast(
            Sequence[imitation.data.types.TrajectoryWithRew], trajectories_dataset_sequence
        )
        return episode_sequence

    @staticmethod
    def from_dataset(file_path: str) -> "EpisodeSequence":
        """
        Initialize an EpisodeSequence based on provided huggingface dataset path.

        Episode sequences are loaded from a provided file path in the agent section of the config.
        Files of recorded episode sequences are generated by saving a sequence of `imitation.TrajectoryWithRew` objects.
        https://imitation.readthedocs.io/en/latest/main-concepts/trajectories.html#storing-loading-trajectories

        Args:
            file_path (str): Path to huggingface dataset recording of episodes.

        Returns:
            episode_sequence: Representation of episode sequence (this class).
        """

        episode_sequence = EpisodeSequence()
        trajectories_dataset_sequence = serialize.load(file_path)
        episode_sequence._episodes = cast(
            Sequence[imitation.data.types.TrajectoryWithRew], trajectories_dataset_sequence
        )
        return episode_sequence

    def save(self, file_path):
        """
        Save episode sequence into a file, saved as HuggingFace dataset.
        To load these episodes again, you can call the `.from_dataset` method.

        Args:
            file_path: File path and file name to save episode sequence to.
        """
        serialize.save(file_path, self._episodes)

    # FIXME: Since EpisodeSequence is now memory-efficient, using SizedGenerator does not make sense anymore.
    #   => Adjust agents to take EpisodeSequence instead of SizedGenerator.

    def generate_episodes(self):
        while True:
            for episode in self:
                yield episode

    def to_imitation_episodes(self) -> SizedGenerator[imitation.data.types.TrajectoryWithRew]:
        return SizedGenerator(self.generate_episodes(), len(self), True)

    def to_d3rlpy_episodes(self) -> SizedGenerator[d3rlpy.dataset.components.Episode]:
        class WrappedEpisodeSequence(EpisodeSequence):
            def __init__(self, episodes: Sequence[imitation.data.types.TrajectoryWithRew]):
                super().__init__()
                self._episodes = episodes

            def __getitem__(self, index):
                e = self._episodes[index]
                generic_episode = generate_generic_episode_from_imitation_trajectory(e)
                return generate_d3rlpy_episode_from_generic_episode(generic_episode)

        episode_sequence = WrappedEpisodeSequence(self._episodes)
        return SizedGenerator(episode_sequence.generate_episodes(), len(episode_sequence), True)

    def to_generic_episodes(self) -> SizedGenerator[GenericEpisode]:
        class WrappedEpisodeSequence(EpisodeSequence):
            def __init__(self, episodes: Sequence[imitation.data.types.TrajectoryWithRew]):
                super().__init__()
                self._episodes = episodes

            def __getitem__(self, index):
                e = self._episodes[index]
                generic_episode = generate_generic_episode_from_imitation_trajectory(e)
                return generic_episode

        episode_sequence = WrappedEpisodeSequence(self._episodes)
        return SizedGenerator(episode_sequence.generate_episodes(), len(episode_sequence), True)

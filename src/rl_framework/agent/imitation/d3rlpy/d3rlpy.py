import io
import logging
import math
import pickle
from functools import partial
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Sequence, Type, Union

import d3rlpy.dataset
import gymnasium as gym
import numpy as np
import pettingzoo
import supersuit as ss
from d3rlpy import LoggingStrategy
from d3rlpy.algos import (
    AWAC,
    BC,
    BCQ,
    BEAR,
    CQL,
    CRR,
    DDPG,
    DQN,
    IQL,
    NFQ,
    PLAS,
    SAC,
    TD3,
    AWACConfig,
    BCConfig,
    BCQConfig,
    BEARConfig,
    CalQL,
    CalQLConfig,
    CQLConfig,
    CRRConfig,
    DDPGConfig,
    DecisionTransformer,
    DecisionTransformerConfig,
    DiscreteBC,
    DiscreteBCConfig,
    DiscreteBCQ,
    DiscreteBCQConfig,
    DiscreteCQL,
    DiscreteCQLConfig,
    DiscreteDecisionTransformer,
    DiscreteDecisionTransformerConfig,
    DiscreteRandomPolicy,
    DiscreteRandomPolicyConfig,
    DiscreteSAC,
    DiscreteSACConfig,
    DoubleDQN,
    DoubleDQNConfig,
    DQNConfig,
    IQLConfig,
    NFQConfig,
    PLASConfig,
    PLASWithPerturbation,
    PLASWithPerturbationConfig,
    RandomPolicy,
    RandomPolicyConfig,
    ReBRAC,
    ReBRACConfig,
    SACConfig,
    TD3Config,
    TD3PlusBC,
    TD3PlusBCConfig,
)
from d3rlpy.base import LearnableBase, LearnableConfig, LearnableConfigWithShape
from d3rlpy.dataset import (
    BufferProtocol,
    PartialTrajectory,
    ReplayBuffer,
    Transition,
    dump,
)
from d3rlpy.logging import LOG, LoggerAdapter, LoggerAdapterFactory
from d3rlpy.types import NDArray, Observation
from stable_baselines3.common.env_util import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from rl_framework.agent.imitation.episode_sequence import EpisodeSequence
from rl_framework.agent.imitation_learning_agent import Agent, ILAgent
from rl_framework.util import (
    Connector,
    DummyConnector,
    FeaturesExtractor,
    SizedGenerator,
    patch_d3rlpy,
    wrap_environment_with_features_extractor_preprocessor,
)

patch_d3rlpy()

D3RLPY_ALGORITHM_CONFIG_REGISTRY: dict[Type[LearnableBase], Type[LearnableConfig]] = {
    AWAC: AWACConfig,
    BC: BCConfig,
    BCQ: BCQConfig,
    BEAR: BEARConfig,
    CalQL: CalQLConfig,
    CQL: CQLConfig,
    CRR: CRRConfig,
    DDPG: DDPGConfig,
    DecisionTransformer: DecisionTransformerConfig,
    DoubleDQN: DoubleDQNConfig,
    DQN: DQNConfig,
    IQL: IQLConfig,
    NFQ: NFQConfig,
    PLAS: PLASConfig,
    PLASWithPerturbation: PLASWithPerturbationConfig,
    ReBRAC: ReBRACConfig,
    SAC: SACConfig,
    TD3: TD3Config,
    TD3PlusBC: TD3PlusBCConfig,
    RandomPolicy: RandomPolicyConfig,
    DiscreteBC: DiscreteBCConfig,
    DiscreteBCQ: DiscreteBCQConfig,
    DiscreteCQL: DiscreteCQLConfig,
    DiscreteDecisionTransformer: DiscreteDecisionTransformerConfig,
    DiscreteSAC: DiscreteSACConfig,
    DiscreteRandomPolicy: DiscreteRandomPolicyConfig,
}


class ConnectorAdapterFactory(LoggerAdapterFactory):
    def __init__(self, connector: Connector, agent: Agent):
        self.connector = connector
        self.agent = agent

    def create(self, experiment_name: str) -> LoggerAdapter:
        return ConnectorAdapter(self.connector, self.agent)


class ConnectorAdapter(LoggerAdapter):
    """
    This class uploads metrics and models to experiment tracking services using a Connector.

    Args:
        connector (Connector): Connector which establishes connection to experiment tracking service.
    """

    def __init__(self, connector: Connector, agent: Agent):
        self.connector = connector
        self.agent = agent

    def write_params(self, params: Dict[str, Any]) -> None:
        pass

    def before_write_metric(self, epoch: int, step: int) -> None:
        pass

    def write_metric(self, epoch: int, step: int, name: str, value: float) -> None:
        absolute_step = step * self.agent.algorithm._config.batch_size
        self.connector.log_value_with_timestep(absolute_step, value, name)

    def after_write_metric(self, epoch: int, step: int) -> None:
        pass

    def save_model(self, epoch, algo) -> None:
        # NOTE: `epoch` is actually `batch_number` when this is called from the algorithm's train method.
        self.connector.upload(agent=self.agent, checkpoint_id=epoch)

    def close(self) -> None:
        pass


class GeneratorReplayBuffer(ReplayBuffer):
    """
    This replay buffer is a memory-efficient buffer that is initialized with a generator of episodes.
    The main down-side is that "random access" of transitions and trajectories is not random,
    but comes in the order of episodes in the generator.
    """

    def __init__(
        self,
        episode_generator: Optional[SizedGenerator[d3rlpy.dataset.Episode]],
        episodes: Optional[Sequence[d3rlpy.dataset.Episode]] = None,
        env: Optional[gym.Env] = None,
        *args,
        **kwargs,
    ):
        if not env:
            raise BufferError(
                "Environment must be provided to the GeneratorReplayBuffer to determine "
                "observation/action/reward properties."
            )

        self._transition_count_cached = None

        if episode_generator:
            self._episode_generator = episode_generator
        elif episodes:

            def generator_from_sequence(sequence):
                for elem in sequence:
                    yield elem

            self._episode_generator = SizedGenerator(
                generator_from_sequence(episodes), size=len(episodes), looping=True
            )
        else:
            raise BufferError("Episodes must be provided to the GeneratorReplayBuffer (later appending not possible).")

        self._current_episode: d3rlpy.dataset.Episode = next(self._episode_generator)
        self._current_transition_index: int = 0

        class NotImplementedBuffer(BufferProtocol):
            pass

        # self._buffer should never be used. Passing a non-implemented buffer to raise an error if usage is attempted.
        super().__init__(*args, buffer=NotImplementedBuffer(), episodes=None, env=env, **kwargs)

    def cycle_episode(self):
        self._current_episode = next(self._episode_generator)
        self._current_transition_index = 0

    def cycle_transition(self):
        self._current_transition_index += 1
        if self._current_transition_index > self._current_episode.transition_count - 1:
            self.cycle_episode()

    def append(
        self,
        observation: Observation,
        action: Union[int, NDArray],
        reward: Union[float, NDArray],
    ) -> None:
        raise BufferError("This buffer should not be extended. It is read-only.")

    def append_episode(self, episode: d3rlpy.dataset.Episode) -> None:
        raise BufferError("This buffer should not be extended. It is read-only.")

    def clip_episode(self, terminated: bool) -> None:
        raise BufferError("This buffer should not be extended. It is read-only.")

    def sample_transition(self) -> Transition:
        transition = self._transition_picker(self._current_episode, self._current_transition_index)
        self.cycle_transition()
        return transition

    def sample_trajectory(self, length: int) -> PartialTrajectory:
        trajectory_end_index = np.random.randint(self._current_episode.transition_count)
        trajectory = self._trajectory_slicer(self._current_episode, trajectory_end_index, length)
        self.cycle_episode()
        return trajectory

    def dump(self, f: BinaryIO) -> None:
        dump(self.episodes, f)

    @property
    def episodes(self) -> Sequence[d3rlpy.dataset.Episode]:
        LOG.warning("This operation is discouraged. Loading episodes into memory from memory-efficient buffer.")
        episodes = []
        for _ in range(len(self._episode_generator)):
            episode = next(self._episode_generator)
            episodes.append(episode)
        return episodes

    def size(self) -> int:
        return len(self._episode_generator)

    @property
    def transition_count(self) -> int:
        LOG.warning("Usage of transition_count is discouraged. Cycling through all episodes for calculation.")
        transition_count = 0
        for _ in range(len(self._episode_generator)):
            episode = next(self._episode_generator)
            transition_count += episode.transition_count
        return transition_count


class D3RLPYAgent(ILAgent):
    @property
    def algorithm(self) -> LearnableBase:
        return self._algorithm

    @algorithm.setter
    def algorithm(self, value: LearnableBase):
        self._algorithm = value

    def __init__(
        self,
        algorithm_class: Type[LearnableBase] = DQN,
        algorithm_parameters: Optional[Dict] = None,
        features_extractor: Optional[FeaturesExtractor] = None,
    ):
        """
        Initialize an agent which will trained on one of imitation algorithms.

        Args:
            algorithm_class (Type[LearnableBase]): RL algorithm class. Specifies the algorithm for RL training.
                Defaults to DQN.
            algorithm_parameters (Dict): Parameters / keyword arguments for the specified offline RL algorithm class.
                - "device" (str): Device to use for training. Defaults to None.
                - See https://d3rlpy.readthedocs.io/en/v2.6.0/references/algos.html for details on individual params.
        """
        super().__init__(algorithm_class, algorithm_parameters, features_extractor)
        device = self.algorithm_parameters.pop("device", None)
        config = D3RLPY_ALGORITHM_CONFIG_REGISTRY[self.algorithm_class](**self.algorithm_parameters)
        self.algorithm = config.create(device=device)
        if device:
            self.algorithm_parameters.update({"device": device})

    def train(
        self,
        total_timesteps: int,
        episode_sequence: EpisodeSequence,
        validation_episode_sequence: Optional[EpisodeSequence] = None,
        connector: Optional[Connector] = None,
        training_environments: Optional[List[Union[gym.Env, pettingzoo.ParallelEnv]]] = None,
        *args,
        **kwargs,
    ):
        """
        Train the instantiated agent on a list of trajectories.

        This training is done by using imitation learning policies, provided by the imitation library.

        The model is changed in place, therefore the updated model can be accessed in the `.model` attribute
        after the agent has been trained.

        Args:
            total_timesteps (int): Amount of (recorded) timesteps to train the agent on.
            episode_sequence (EpisodeSequence): List of episodes on which the agent should be trained on.
            validation_episode_sequence (EpisodeSequence): List of episodes on which the agent should be validated on.
            training_environments (List): List of environments with gym (or pettingzoo) interface.
                Required for callbacks and logging. The environments are used for intermediate rollouts.
                Also required for attribute extraction (e.g., action/observation space).
            connector (Connector): Connector for executing callbacks (e.g., logging metrics and saving checkpoints)
                on training time. Calls need to be declared manually in the code.
        """

        assert len(training_environments) > 0, "All offline RL algorithms require an environment to be passed. "

        def make_env(index: int):
            return training_environments[index]

        if not episode_sequence:
            raise ValueError("No transitions have been provided to the train-method.")

        if not connector:
            connector = DummyConnector()

        trajectories: SizedGenerator[d3rlpy.dataset.Episode] = episode_sequence.to_d3rlpy_episodes()
        validation_trajectories: SizedGenerator[d3rlpy.dataset.Episode] = (
            validation_episode_sequence.to_d3rlpy_episodes() if validation_episode_sequence else None
        )

        if self.features_extractor:

            def preprocess_d3rlpy_episodes(trajectories):
                for trajectory in trajectories:
                    yield d3rlpy.dataset.Episode(
                        observations=self.features_extractor.preprocess(trajectory.observations),
                        actions=trajectory.actions,
                        rewards=trajectory.rewards,
                        terminated=trajectory.terminated,
                    )

            trajectories = SizedGenerator(
                preprocess_d3rlpy_episodes(trajectories), size=len(trajectories), looping=trajectories.looping
            )
            validation_trajectories = (
                SizedGenerator(
                    preprocess_d3rlpy_episodes(validation_trajectories),
                    size=len(validation_trajectories),
                    looping=validation_trajectories.looping,
                )
                if validation_trajectories
                else None
            )

            training_environments = [
                wrap_environment_with_features_extractor_preprocessor(env, self.features_extractor)
                for env in training_environments
            ]

        if isinstance(training_environments[0], pettingzoo.ParallelEnv):

            def pettingzoo_environment_to_vectorized_environment(pettingzoo_environment: pettingzoo.ParallelEnv):
                env = ss.black_death_v3(pettingzoo_environment)
                env = ss.pettingzoo_env_to_vec_env_v1(env)
                env = ss.concat_vec_envs_v1(env, 1, num_cpus=1, base_class="stable_baselines3")
                return env

            if len(training_environments) > 1:
                logging.warning(
                    f"Offline Reinforcement Learning algorithm {self.__class__.__qualname__} does not support "
                    f"training on multiple multi-agent environments in parallel. Continuing with one environment as "
                    f"training environment."
                )

            vectorized_environment = pettingzoo_environment_to_vectorized_environment(training_environments[0])
        else:
            training_environments = [Monitor(env) for env in training_environments]
            environment_return_functions = [
                partial(make_env, env_index) for env_index in range(len(training_environments))
            ]
            vectorized_environment = self.to_vectorized_env(env_fns=environment_return_functions)

        replay_buffer = GeneratorReplayBuffer(episode_generator=trajectories, env=training_environments[0])

        evaluators = {
            "episode_reward_mean": d3rlpy.metrics.EnvironmentEvaluator(training_environments[0]),
        }
        if validation_trajectories:
            validation_buffer = GeneratorReplayBuffer(
                episode_generator=validation_trajectories, env=training_environments[0]
            )
            evaluators.update({"td_error": d3rlpy.metrics.TDErrorEvaluator(validation_buffer.episodes)})

        n_batches = math.ceil(total_timesteps / self.algorithm._config.batch_size)

        self.algorithm.fit(
            replay_buffer,
            n_steps=n_batches,
            n_steps_per_epoch=min(n_batches, math.ceil(50000 / self.algorithm._config.batch_size)),
            logging_steps=math.ceil(10000 / self.algorithm._config.batch_size),
            logging_strategy=LoggingStrategy.STEPS,
            logger_adapter=ConnectorAdapterFactory(connector, self),
            save_interval=1,
            evaluators=evaluators,
        )

        # FIXME: Check case of continuous training (works, but deteriorates for small number of fine-tuning steps)
        # FIXME: Check when LOG.warn for GeneratorReplayBuffer is triggered (loading in-memory)

        vectorized_environment.close()

    @staticmethod
    def to_vectorized_env(env_fns) -> VecEnv:
        return SubprocVecEnv(env_fns)

    def choose_action(self, observation: object, deterministic: bool = False, *args, **kwargs):
        """
        Chooses action which the agent will perform next, according to the observed environment.

        Args:
            observation (object): Observation of the environment
            deterministic (bool): Whether the action should be determined in a deterministic or stochastic way.

        Returns: action (int): Action to take according to policy.

        """

        if not self.algorithm:
            raise ValueError("Cannot predict action for uninitialized agent. Start a training first to initialize.")

        action = self.algorithm.predict(
            np.expand_dims(observation, axis=0),
        )[0]
        return action

    def save_to_file(self, file_path: Path, *args, **kwargs) -> None:
        """Save the agent to a file (for later loading).

        Args:
            file_path (Path): The path where the agent should be saved to.
        """
        self.algorithm.save(file_path.as_posix())

    def load_from_file(self, file_path: Path, algorithm_parameters: Dict = None, *args, **kwargs) -> None:
        """Loads the agent policy (`self.agent_policy`) in-place from a zipped folder.
        The agent algorithm (`self.algorithm`) is re-initialized in the train method and remains None until then.

        Args:
            file_path (Path): The file path the agent has been saved to before.
            algorithm_parameters: Parameters to be set for the loaded algorithm.
                Providing None leads to keeping the previously set parameters.
        """
        if algorithm_parameters:
            self.algorithm_parameters = algorithm_parameters

        with open(file_path, "rb") as f:
            obj = pickle.load(f)

        device = self.algorithm_parameters.pop("device", None)
        config = LearnableConfigWithShape.deserialize(obj["config"])
        config.config = D3RLPY_ALGORITHM_CONFIG_REGISTRY[self.algorithm_class](**self.algorithm_parameters)
        self.algorithm = config.create(device=device)
        if device:
            self.algorithm_parameters.update({"device": device})

        self.algorithm.impl.load_model(io.BytesIO(obj["torch"]))

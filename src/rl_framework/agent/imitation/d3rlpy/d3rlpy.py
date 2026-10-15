import io
import logging
import math
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import d3rlpy.dataset
import gymnasium as gym
import numpy as np
import pettingzoo
from d3rlpy import ActionSpace, LoggingStrategy
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
    TransformerAlgoBase,
)
from d3rlpy.base import LearnableBase, LearnableConfig, LearnableConfigWithShape
from d3rlpy.dataset import ReplayBuffer
from d3rlpy.logging import LoggerAdapter, LoggerAdapterFactory
from stable_baselines3.common.vec_env import VecEnv

from rl_framework.agent.imitation.episode_sequence import (
    EpisodeSequence,
    WrappedEpisodeSequence,
)
from rl_framework.agent.imitation_learning_agent import Agent, ILAgent
from rl_framework.util import (
    Connector,
    DummyConnector,
    Environment,
    FeaturesExtractor,
    patch_d3rlpy_map_location,
    patch_d3rlpy_onnx_export,
    wrap_environment_with_features_extractor_preprocessor,
)
from rl_framework.util.onnx_export import export_d3rlpy_algorithm_as_onnx

patch_d3rlpy_map_location()
patch_d3rlpy_onnx_export()

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

# Algorithms which do not estimate values (their implementation cannot compute e.g. the TD error)
VALUE_FREE_ALGORITHMS = (BC, DiscreteBC, RandomPolicy, DiscreteRandomPolicy)


class _SpacesEnvironment(gym.Env):
    """
    Observation and action space of a training environment, for d3rlpy which detects them from a gym environment.
    It cannot be stepped.
        - gym.Env and VecEnv: their spaces (a VecEnv's spaces are those of each sub-environment).
        - PettingZoo environment: the spaces of its first agent (the agents share one policy).
        - EnvironmentFactory (stub_environment, env_return_function): the spaces of the stub environment (the
          environment itself is not created).
    """

    def __init__(self, environment: Environment):
        if isinstance(environment, pettingzoo.ParallelEnv):
            agent = environment.possible_agents[0]
            self.observation_space = environment.observation_space(agent)
            self.action_space = environment.action_space(agent)
        elif isinstance(environment, (gym.Env, VecEnv, tuple)):
            spaces_source = environment[0] if isinstance(environment, tuple) else environment
            self.observation_space = spaces_source.observation_space
            self.action_space = spaces_source.action_space
        else:
            raise TypeError(
                f"Training environment of unsupported type {type(environment)} provided to offline reinforcement "
                f"learning agent."
            )


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
        # Action space of the training environment (d3rlpy only knows the action size): continuous actions are
        #   clipped to it, like SB3's `predict`
        self.action_space: Optional[gym.Space] = None

    def train(
        self,
        total_timesteps: int,
        episode_sequence: EpisodeSequence,
        validation_episode_sequence: Optional[EpisodeSequence] = None,
        connector: Optional[Connector] = None,
        training_environments: Optional[Environment] = None,
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
            training_environments (List): List of environments: gym.Env, VecEnv, PettingZoo ParallelEnv or
                EnvironmentFactory (stub_environment, env_return_function).
                The first one provides the observation and action space. Except for environment factories, the
                environments are also used for the online evaluation after every epoch (`episode_reward_mean`, the
                mean reward of 10 deterministic episodes, see `Agent.evaluate`); they are not closed.
            connector (Connector): Connector for executing callbacks (e.g., logging metrics and saving checkpoints)
                on training time. Calls need to be declared manually in the code.
        """

        assert len(training_environments) > 0, "All offline RL algorithms require an environment to be passed. "

        if not episode_sequence:
            raise ValueError("No transitions have been provided to the train-method.")

        if not connector:
            connector = DummyConnector()

        # Unwrapped: `evaluate` preprocesses the observations with the features extractor itself
        evaluation_environments = list(training_environments)

        trajectories: WrappedEpisodeSequence[d3rlpy.dataset.Episode] = episode_sequence.to_d3rlpy_episodes()
        validation_trajectories: WrappedEpisodeSequence[d3rlpy.dataset.Episode] = (
            validation_episode_sequence.to_d3rlpy_episodes() if validation_episode_sequence else None
        )

        if self.features_extractor:
            """
            Convert an Episode to an Episode with preprocessed observations.
            """

            def preprocess_observations_with_features_extractor(
                d3rlpy_episode: d3rlpy.dataset.Episode,
            ) -> d3rlpy.dataset.Episode:
                return d3rlpy.dataset.Episode(
                    observations=self.features_extractor.preprocess(d3rlpy_episode.observations),
                    actions=d3rlpy_episode.actions,
                    rewards=d3rlpy_episode.rewards,
                    terminated=d3rlpy_episode.terminated,
                )

            trajectories = trajectories.episode_sequence_from_additional_conversion(
                preprocess_observations_with_features_extractor
            )
            validation_trajectories = (
                validation_trajectories.episode_sequence_from_additional_conversion(
                    preprocess_observations_with_features_extractor
                )
                if validation_trajectories
                else None
            )

            training_environments = [
                wrap_environment_with_features_extractor_preprocessor(env, self.features_extractor)
                for env in training_environments
            ]

        spaces_environment = _SpacesEnvironment(training_environments[0])
        self.action_space = spaces_environment.action_space
        if isinstance(training_environments[0], tuple):
            # Creating the environments of the factories after every epoch would be too expensive
            evaluation_environments = None
            logging.warning(
                "Offline Reinforcement Learning on environment factories: the stub environment only provides the "
                "spaces, the environments are not used for online evaluation during training (episode_reward_mean is "
                "not logged)."
            )
        is_transformer = isinstance(self.algorithm, TransformerAlgoBase)
        if is_transformer:
            # FIXME: `choose_action` cannot run decision transformers (they need the context of the episode and its
            #   rewards, like d3rlpy's `StatefulTransformerWrapper`), so `evaluate` and the validation evaluators (which
            #   call `predict` with single observations) fail for them
            evaluation_environments, validation_trajectories = None, None
            logging.warning(
                "Decision transformers are neither evaluated online nor on validation episodes during training "
                "(episode_reward_mean and the validation metrics are not logged): the agent does not support it yet."
            )

        # Action space and size passed explicitly: d3rlpy's `ReplayBuffer` only detects them from `env` if no
        #   episodes are given, otherwise from the episodes, i.e. the action size from the highest recorded action
        #   (max + 1), which is too small if the demonstrations never take the highest action
        spaces = {
            "env": spaces_environment,
            "action_space": d3rlpy.dataset.detect_action_space_from_env(spaces_environment),
            "action_size": d3rlpy.dataset.detect_action_size_from_env(spaces_environment),
        }
        replay_buffer = ReplayBuffer(buffer=d3rlpy.dataset.InfiniteBuffer(), episodes=trajectories, **spaces)

        evaluators = {}
        if evaluation_environments is not None:
            evaluators["episode_reward_mean"] = self._evaluate_online_on(evaluation_environments)
        if validation_trajectories:
            validation_buffer = ReplayBuffer(
                buffer=d3rlpy.dataset.InfiniteBuffer(), episodes=validation_trajectories, **spaces
            )
            evaluators.update(self._get_validation_evaluators(validation_buffer.episodes))

        batch_size = self.algorithm._config.batch_size
        n_batches = math.ceil(total_timesteps / batch_size)
        n_steps_per_epoch = min(n_batches, math.ceil(50000 / batch_size))
        # At least one logging step per epoch (d3rlpy's fitter fails otherwise)
        logging_steps = min(n_steps_per_epoch, math.ceil(10000 / batch_size))

        def evaluate_after_epoch(algorithm: LearnableBase, epoch: int, total_step: int) -> None:
            # Evaluated and logged here instead of passing the evaluators to `fit`: with LoggingStrategy.STEPS, d3rlpy
            #   adds evaluator scores only after the epoch's last logging step, so they would be logged an epoch late
            #   (and never for the last epoch).
            for name, evaluator in evaluators.items():
                connector.log_value_with_timestep(total_step * batch_size, evaluator(algorithm, replay_buffer), name)

        if is_transformer:
            # d3rlpy's `fit` of decision transformers logs once per epoch and has no epoch callback (no evaluators)
            self.algorithm.fit(
                replay_buffer,
                n_steps=n_batches,
                n_steps_per_epoch=n_steps_per_epoch,
                logger_adapter=ConnectorAdapterFactory(connector, self),
                save_interval=1,
            )
        else:
            self.algorithm.fit(
                replay_buffer,
                n_steps=n_batches,
                n_steps_per_epoch=n_steps_per_epoch,
                logging_steps=logging_steps,
                logging_strategy=LoggingStrategy.STEPS,
                logger_adapter=ConnectorAdapterFactory(connector, self),
                save_interval=1,
                epoch_callback=evaluate_after_epoch,
            )

        # FIXME: Check case of continuous training (works, but deteriorates for small number of fine-tuning steps)
        # FIXME: Check when LOG.warn for GeneratorReplayBuffer is triggered (loading in-memory)

    def _evaluate_online_on(self, evaluation_environments: List[Environment]):
        """
        Evaluator (called like d3rlpy's evaluators) of the mean reward of 10 deterministic episodes in the
        environments, with the framework's `evaluate` (supports gym, VecEnv and PettingZoo environments, unlike
        d3rlpy's `EnvironmentEvaluator`). The environments are kept open for the next evaluation.
        """

        def evaluate_online(algorithm: LearnableBase, dataset: ReplayBuffer) -> float:
            mean_reward, _ = self.evaluate(
                evaluation_environments,
                n_eval_episodes=10,
                connector=DummyConnector(),
                deterministic=True,
                close_environments=False,
            )
            return float(mean_reward)

        return evaluate_online

    def _get_validation_evaluators(self, validation_episodes) -> Dict[str, Any]:
        """
        Evaluators on validation episodes: how well the agent's actions match the recorded ones, and the TD error for
        algorithms which estimate values (value-free algorithms like (Discrete)BC cannot compute it).
        """
        if self.algorithm.get_action_type() == ActionSpace.DISCRETE:
            evaluators = {"action_match": d3rlpy.metrics.DiscreteActionMatchEvaluator(validation_episodes)}
        else:
            evaluators = {"action_difference": d3rlpy.metrics.ContinuousActionDiffEvaluator(validation_episodes)}
        if not isinstance(self.algorithm, VALUE_FREE_ALGORITHMS):
            evaluators["td_error"] = d3rlpy.metrics.TDErrorEvaluator(validation_episodes)
        return evaluators

    def choose_action(self, observation: object, deterministic: bool = False, *args, **kwargs):
        """
        Chooses action which the agent will perform next, according to the observed environment.

        Args:
            observation (object): Observation of the environment
            deterministic (bool): Whether the action should be determined in a deterministic or stochastic way.

        Returns: action (int): Action to take according to policy.

        """

        if isinstance(self.algorithm, TransformerAlgoBase):
            raise NotImplementedError(
                "Decision transformers cannot choose actions from single observations: they need the context of the "
                "episode and its rewards (see d3rlpy's `StatefulTransformerWrapper`, or the ONNX export)."
            )
        # The algorithm is created on construction, its networks (`impl`) only by training or loading
        if self.algorithm.impl is None:
            raise ValueError("Cannot predict action for uninitialized agent. Start a training first to initialize.")

        action = self.algorithm.predict(
            np.expand_dims(observation, axis=0),
        )[0]
        if isinstance(self.action_space, gym.spaces.Box):
            action = np.clip(action, self.action_space.low, self.action_space.high)
        return action

    def save_policy_as_onnx(self, file_path: Path, deterministic: bool = True, *args, **kwargs) -> None:
        """Save the agent's policy as ONNX model, in the contract of `StableBaselinesAgent.save_policy_as_onnx` (inputs
        and outputs: see `rl_framework.util.onnx_export.export_d3rlpy_algorithm_as_onnx`; decision transformers are
        exported one step per call, with further inputs).

        Args:
            file_path (Path): The file where the policy should be saved to.
            deterministic (bool): Whether the most likely action is exported (default) or, for discrete decision
                transformers only, one sampled from noise inputs.
        """
        if self.action_space is None and self.algorithm.get_action_type() == ActionSpace.CONTINUOUS:
            logging.warning(
                "The action space is unknown (agent saved without it): continuous actions of the ONNX export are "
                "not clipped."
            )
        export_d3rlpy_algorithm_as_onnx(
            self.algorithm, file_path, deterministic=deterministic, action_space=self.action_space
        )

    def save_to_file(self, file_path: Path, *args, **kwargs) -> None:
        """Save the agent to a file (for later loading).

        Args:
            file_path (Path): The path where the agent should be saved to.
        """
        self.algorithm.save(Path(file_path).as_posix())
        if self.action_space is not None:
            # Next to d3rlpy's keys ("torch", "config", "version"), the only ones its `load_learnable` reads
            with open(file_path, "rb") as f:
                obj = pickle.load(f)
            obj["action_space"] = self.action_space
            with open(file_path, "wb") as f:
                pickle.dump(obj, f)

    def load_from_file(self, file_path: Path, algorithm_parameters: Dict = None, *args, **kwargs) -> None:
        """Loads the agent policy (`self.agent_policy`) in-place from a zipped folder.
        The agent algorithm (`self.algorithm`) is re-initialized in the train method and remains None until then.

        Args:
            file_path (Path): The file path the agent has been saved to before.
            algorithm_parameters: Parameters to be set for the loaded algorithm.
                Providing None leads to keeping the previously set parameters.
        """
        if algorithm_parameters:
            self.algorithm_parameters = {**algorithm_parameters}

        with open(file_path, "rb") as f:
            obj = pickle.load(f)

        device = self.algorithm_parameters.pop("device", None)
        config = LearnableConfigWithShape.deserialize(obj["config"])
        algorithm_config = D3RLPY_ALGORITHM_CONFIG_REGISTRY[self.algorithm_class](**self.algorithm_parameters)
        # The scalers are fitted in training (to the dataset or environment): the stored ones, unless fitted ones are
        #   given (the parameters usually hold unfitted ones, or none)
        for scaler_name in ("observation_scaler", "action_scaler", "reward_scaler"):
            scaler = getattr(algorithm_config, scaler_name)
            if scaler is None or not scaler.built:
                setattr(algorithm_config, scaler_name, getattr(config.config, scaler_name))
        config.config = algorithm_config
        self.algorithm = config.create(device=device)
        if device:
            self.algorithm_parameters.update({"device": device})

        self.algorithm.impl.load_model(io.BytesIO(obj["torch"]))
        self.action_space = obj.get("action_space")

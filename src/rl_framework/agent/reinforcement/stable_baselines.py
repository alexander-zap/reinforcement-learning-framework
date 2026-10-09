import logging
import tempfile
from functools import partial
from os import cpu_count
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Type

import gymnasium
import numpy as np
import pettingzoo
import stable_baselines3
import torch
import torch.onnx
from stable_baselines3.common.base_class import BaseAlgorithm, BasePolicy
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.env_util import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecEnv, VecMonitor
from supersuit.vector import MakeCPUAsyncConstructor, MarkovVectorEnv
from supersuit.vector.sb3_vector_wrapper import SB3VecEnvWrapper

from rl_framework.agent.reinforcement_learning_agent import RLAgent
from rl_framework.util import (
    Connector,
    DummyConnector,
    Environment,
    FeaturesExtractor,
    GammaScheduleCallback,
    LoggingCallback,
    ResetInfoCallback,
    SavingCallback,
    apply_action_bias,
    check_saved_features_extractor,
    check_specified_policy_kwargs,
    get_sb3_policy_kwargs_for_features_extractor,
    reset_optimizer_state,
    validate_initial_action_bias,
    wrap_environment_with_features_extractor_preprocessor,
)


class StableBaselinesAgent(RLAgent):
    @property
    def algorithm(self) -> BaseAlgorithm:
        return self._algorithm

    @algorithm.setter
    def algorithm(self, value: BaseAlgorithm):
        self._algorithm = value

    def __init__(
        self,
        algorithm_class: Type[BaseAlgorithm] = stable_baselines3.PPO,
        algorithm_parameters: Optional[Dict] = None,
        features_extractor: Optional[FeaturesExtractor] = None,
    ):
        """
        Initialize an agent which will trained on one of Stable-Baselines3 algorithms.

        Args:
            algorithm_class (Type[BaseAlgorithm]): SB3 RL algorithm class. Specifies the algorithm for RL training.
                Defaults to PPO.
            algorithm_parameters (Dict): Parameters / keyword arguments for the specified SB3 RL Algorithm class.
                See https://stable-baselines3.readthedocs.io/en/master/modules/base.html for details on common params.
                See individual docs (e.g., https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
                for algorithm-specific params.
                `policy` defaults to "MlpPolicy", `tensorboard_log` to a new temporary directory.
                Additionally, these framework parameters are applied by the agent (and not passed to SB3):
                    - `gamma` may be a callable `progress_remaining -> gamma` (like SB3's `learning_rate`) to schedule
                      the discount factor over training.
                    - `initial_action_bias`: initial biases of the policy's action output layer (PPO, A2C, TRPO, SAC,
                      TD3), applied to a freshly created model (see `rl_framework.util.policy_init`).
                    - `reset_optimizer` (bool, default False): reset the optimizer state before each training, e.g.,
                      to fine-tune a loaded model with fresh optimizer statistics.
                    - `callback_kwargs` (Dict): parameters of the training callbacks: `callback_saving_interval`
                      (default 500000 steps), `callback_logging_interval` (default 1 episode),
                      `callback_log_distributions` (default False), `callback_verbosity` (default 0),
                      `sb3_logging_interval` (default 1) and, for AsyncStableBaselinesAgent,
                      `callback_async_utilization_logging_interval` (default 1000 episodes).
                The same applies to the parameters given to `load_from_file` (or `download`), which replace these.
            features_extractor: When provided, specifies the observation processor to be
                    used before the action/value prediction network.
        """
        super().__init__(algorithm_class, algorithm_parameters, features_extractor)

        self.algorithm_parameters = self._add_required_default_parameters(self.algorithm_parameters)
        self.callback_parameters: Dict = {}
        self.reset_optimizer: bool = False
        self.gamma_schedule: Optional[Callable[[float], float]] = None
        self.initial_action_bias: Optional[np.ndarray] = None
        self._setup_framework_parameters()

        additional_parameters = (
            {"_init_setup_model": False} if (getattr(self.algorithm_class, "_setup_model", None)) else {}
        )

        self.algorithm: BaseAlgorithm = self.algorithm_class(
            env=None, **self.algorithm_parameters, **additional_parameters
        )
        self.algorithm_needs_initialization = True

    def train(
        self,
        total_timesteps: int = 100000,
        connector: Optional[Connector] = None,
        training_environments: List[Environment] = None,
        *args,
        **kwargs,
    ):
        """
        Train the instantiated agent on the environment.

        This training is done by using the agent-on-environment training method provided by Stable-baselines3.

        The model is changed in place, therefore the updated model can be accessed in the `.model` attribute
        after the agent has been trained.

        Args:
            training_environments (List[Environment]): List of environments on which the agent should be trained on.
                PettingZoo ParallelEnvs are reset once all agents are done. Agents which restart on their own while the
                other agents continue follow gymnasium's auto-reset convention: the done step returns the first
                observation of the agent's next episode and passes the last observation of the finished episode as
                `infos[agent]["final_observation"]`. Without this info, the returned observation is used as both.
            total_timesteps (int): Amount of individual steps the agent should take before terminating the training.
            connector (Connector): Connector for executing callbacks (e.g., logging metrics and saving checkpoints)
                on training time. Calls need to be declared manually in the code.
        """

        def make_env(env_list: list, index: int):
            return env_list[index]

        if not training_environments:
            raise ValueError("No training environments have been provided to the train-method.")

        if not connector:
            connector = DummyConnector()

        if self.features_extractor:
            training_environments = [
                wrap_environment_with_features_extractor_preprocessor(env, self.features_extractor)
                for env in training_environments
            ]

        if isinstance(training_environments[0], pettingzoo.ParallelEnv):
            vector_envs = []
            for pettingzoo_environment in training_environments:
                vector_env = MarkovVectorEnv(pettingzoo_environment, black_death=True)
                vector_envs.append(vector_env)

            environment_return_functions = [
                partial(make_env, vector_envs, env_index) for env_index in range(len(vector_envs))
            ]

            vectorized_environment = MakeCPUAsyncConstructor(min(cpu_count(), len(environment_return_functions)))(
                environment_return_functions, vector_envs[0].observation_space, vector_envs[0].action_space
            )

            class AutoResetSB3VecEnvWrapper(SB3VecEnvWrapper):
                """
                A SB3VecEnvWrapper for Pettingzoo based vectorized environments (through MarkovVectorEnv).
                MarkovVectorEnv resets the environment automatically once all agents are done and returns the done
                flags and rewards of the final step together with the observations of the new episode.
                Sets infos for each done agent on step (as SB3 expects from auto-resetting vectorized environments):
                    - `infos["terminal_observation"]`: the last observation of the finished episode
                    - `infos["TimeLimit.truncated"] = True` when truncated (else False)

                For agents which restart on their own, the `final_observation` info (gymnasium's auto-reset convention,
                see `train`) is the last observation of the finished episode. It takes precedence, also over the
                `terminal_observation` which MarkovVectorEnv sets to the returned observation when all agents are done.
                """

                def step_wait(self):
                    observations, rewards, terminations, truncations, infos = self.venv.step_wait()
                    dones = np.array([terminations[i] or truncations[i] for i in range(len(terminations))])
                    for i in np.flatnonzero(dones):
                        infos[i]["TimeLimit.truncated"] = bool(truncations[i] and not terminations[i])
                        if "final_observation" in infos[i]:
                            infos[i]["terminal_observation"] = infos[i]["final_observation"]
                        else:
                            infos[i].setdefault("terminal_observation", observations[i])
                    return observations, rewards, dones, infos

            vectorized_environment = AutoResetSB3VecEnvWrapper(vectorized_environment)
            vectorized_environment = VecMonitor(vectorized_environment)

        elif isinstance(training_environments[0], gymnasium.Env):
            training_environments = [Monitor(env) for env in training_environments]
            environment_return_functions = [
                partial(make_env, training_environments, env_index) for env_index in range(len(training_environments))
            ]

            # noinspection PyCallingNonCallable
            vectorized_environment = self.to_vectorized_env(env_fns=environment_return_functions)

        elif isinstance(training_environments[0], VecEnv):
            assert len(training_environments) == 1
            vectorized_environment = training_environments[0]

        # tuple = EnvironmentFactory in format (stub_environment, env_return_function)
        elif isinstance(training_environments[0], tuple):
            environment_return_functions = []
            stub_environment = None
            for stub_env, env_func in training_environments:
                environment_return_functions.append(env_func)
                stub_environment = stub_env

            # noinspection PyCallingNonCallable
            vectorized_environment = self.to_vectorized_env(
                env_fns=environment_return_functions, stub_env=stub_environment
            )

        else:
            raise TypeError(f"Environment type {type(training_environments[0])} not supported!")

        algorithm_kwargs = {"env": vectorized_environment}
        if self.algorithm_needs_initialization:
            parameters = {**self.algorithm_parameters}
            if self.features_extractor:
                parameters["policy_kwargs"] = self._get_policy_kwargs()
            algorithm_kwargs.update(parameters)
            self.algorithm = self.algorithm_class(**algorithm_kwargs)
            self.algorithm_needs_initialization = False
            # Only a freshly created model gets the initial action bias, before learn() collects its first rollout.
            if self.initial_action_bias is not None:
                apply_action_bias(self.algorithm, self.initial_action_bias)
        else:
            with tempfile.TemporaryDirectory("w") as tmp_dir:
                tmp_path = Path(tmp_dir) / "tmp_model.zip"
                self.save_to_file(tmp_path)
                algorithm_kwargs["path"] = tmp_path
                algorithm_kwargs["custom_objects"] = self._get_parameters_for_loading()
                # noinspection PyUnresolvedReferences
                device = self.algorithm_parameters.get("device", None)
                self.algorithm = (
                    self.algorithm_class.load(**algorithm_kwargs)
                    if not device
                    else self.algorithm_class.load(**algorithm_kwargs, device=device)
                )

        if self.reset_optimizer:
            reset_paths = reset_optimizer_state(self.algorithm)
            logging.info(f"Optimizer state reset for: {', '.join(reset_paths)}")

        callbacks = self.get_callbacks(connector=connector)
        callback_list = CallbackList(callbacks)

        sb3_logging_interval = self.callback_parameters.get("sb3_logging_interval", 1)
        self.algorithm.learn(total_timesteps=total_timesteps, callback=callback_list, log_interval=sb3_logging_interval)
        vectorized_environment.close()

    def to_vectorized_env(self, env_fns, stub_env=None) -> VecEnv:
        return SubprocVecEnv(env_fns)

    def get_callbacks(self, connector: Connector) -> list[BaseCallback]:
        callback_verbosity = self.callback_parameters.get("callback_verbosity", 0)
        callback_saving_interval = self.callback_parameters.get("callback_saving_interval", 500000)
        callback_logging_interval = self.callback_parameters.get("callback_logging_interval", 1)
        callback_log_distributions = self.callback_parameters.get("callback_log_distributions", False)

        callbacks = [
            SavingCallback(
                self, connector=connector, checkpoint_frequency=callback_saving_interval, verbose=callback_verbosity
            ),
            LoggingCallback(
                connector=connector,
                logging_frequency=callback_logging_interval,
                log_distributions=callback_log_distributions,
            ),
            ResetInfoCallback(connector=connector),
        ]
        if self.gamma_schedule is not None:
            callbacks.append(GammaScheduleCallback(self.gamma_schedule, verbose=callback_verbosity))

        return callbacks

    def choose_action(self, observation: object, deterministic: bool = False, *args, **kwargs):
        """
        Chooses action which the agent will perform next, according to the observed environment.

        Args:
            observation (object): Observation of the environment
            deterministic (bool): Whether the action should be determined in a deterministic or stochastic way.

        Returns: action: Action to take according to policy.

        """

        (
            action,
            _,
        ) = self.algorithm.predict(
            observation,
            deterministic=deterministic,
        )
        if not action.shape:
            action = action.item()
        return action

    def save_policy_as_onnx(self, file_path: Path) -> None:
        """Save the policy as ONNX model.

            Details for SB3: https://stable-baselines3.readthedocs.io/en/master/guide/export.html

        Args:
            file_path (Path): The file where the policy should be saved to.
        """
        assert str(file_path).endswith(".onnx"), "File path must end with .onnx"

        class OnnxableSB3Policy(torch.nn.Module):
            def __init__(self, policy: BasePolicy):
                super().__init__()
                self.policy = policy

            # FIXME: policy() returns `actions, values, log_prob` for PPO
            # FIXME: determinism should be set based on policy (and own preference; could be set in config)
            def forward(self, observation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                # NOTE: policy includes the features_extractor.
                #  Preprocessing is included, but postprocessing (clipping/inscaling actions) is not.
                # NOTE: For Box action space you may have to postprocess (unnormalize) actions to the correct bounds
                #   low, high = model.action_space.low, model.action_space.high
                #   post_processed_action = low + (0.5 * (scaled_action + 1.0) * (high - low))
                return self.policy(observation, deterministic=False)

        # FIXME: for algorithms like SAC we need to input `self.algorithm.policy.actor`
        onnx_policy = OnnxableSB3Policy(self.algorithm.policy)
        observation_size = self.algorithm.observation_space.shape
        # FIXME: add support for batch size > 1
        dummy_input = torch.randn(1, *observation_size)
        torch.onnx.export(onnx_policy, dummy_input, file_path, opset_version=17, input_names=["input"])

    def save_to_file(self, file_path: Path, *args, **kwargs) -> None:
        """Save the agent to a file (for later loading).

        Args:
            file_path (Path): The file where the agent should be saved to (SB3 expects a file name ending with .zip).
        """
        self.algorithm.save(file_path)

    def load_from_file(self, file_path: Path, algorithm_parameters: Dict = None, *args, **kwargs) -> None:
        """Load the agent in-place from an agent-save folder.

        Args:
            file_path (Path): The model filename (file ending with .zip).
            algorithm_parameters: Parameters to be set for the loaded algorithm.
                Providing None leads to keeping the previously set parameters.
        """
        if algorithm_parameters:
            self.algorithm_parameters = self._add_required_default_parameters({**algorithm_parameters})
            self._setup_framework_parameters()
        algorithm = self.algorithm_class.load(path=file_path, env=None, **self._get_parameters_for_loading())
        check_saved_features_extractor(algorithm.policy_kwargs, self.features_extractor)
        check_specified_policy_kwargs(algorithm.policy_kwargs, self.algorithm_parameters.get("policy_kwargs") or {})
        self.algorithm = algorithm
        self.algorithm_needs_initialization = False

    def _setup_framework_parameters(self) -> None:
        """
        Move the framework parameters (see `__init__`) out of `algorithm_parameters`, since SB3 does not accept them.
        """
        self.callback_parameters = self.algorithm_parameters.pop("callback_kwargs", {})
        self.reset_optimizer = self.algorithm_parameters.pop("reset_optimizer", False)
        self._setup_gamma_schedule()
        self._setup_initial_action_bias()

    def _get_policy_kwargs(self) -> Dict:
        """
        `policy_kwargs` for creating the model: the user's ones, extended by the features extractor (if provided).
        """
        policy_kwargs = self.algorithm_parameters.get("policy_kwargs") or {}
        if not self.features_extractor:
            return policy_kwargs
        policy = self.algorithm_parameters["policy"]
        policy_class = self.algorithm_class.policy_aliases.get(policy) if isinstance(policy, str) else policy
        return get_sb3_policy_kwargs_for_features_extractor(self.features_extractor, policy_class, policy_kwargs)

    def _get_parameters_for_loading(self) -> Dict:
        """
        Algorithm parameters to apply to a saved model when loading it: all except `policy_kwargs`.
        A saved model keeps its `policy_kwargs` (its architecture; with a features extractor, they contain it), since
        its saved weights only fit them. `check_specified_policy_kwargs` compares the specified ones instead.
        """
        return {key: value for key, value in self.algorithm_parameters.items() if key != "policy_kwargs"}

    def _setup_gamma_schedule(self) -> None:
        """
        Set `gamma_schedule` from `algorithm_parameters`. A callable `gamma` becomes the schedule (applied during
        training by `GammaScheduleCallback`) and is replaced by its start value, since SB3 algorithms require a float.
        A constant (or missing) `gamma` clears the schedule.
        """
        gamma_schedule = self.algorithm_parameters.get("gamma")
        if not callable(gamma_schedule):
            self.gamma_schedule = None
        else:
            self.gamma_schedule = gamma_schedule
            self.algorithm_parameters["gamma"] = float(gamma_schedule(1.0))

    def _setup_initial_action_bias(self) -> None:
        """
        Set `initial_action_bias` from `algorithm_parameters` (see `rl_framework.util.policy_init`), where it is
        removed, since SB3 algorithms do not accept it. It is applied to freshly created models only.
        """
        initial_action_bias = self.algorithm_parameters.pop("initial_action_bias", None)
        self.initial_action_bias = validate_initial_action_bias(initial_action_bias, self.algorithm_class)

    @staticmethod
    def _add_required_default_parameters(algorithm_parameters: Optional[Dict]):
        """
        Add missing required parameters to `algorithm_parameters`.
        Required parameters currently are:
            - "policy": needs to be set for every BaseRLAlgorithm. Set to "MlpPolicy" if not provided.
            - "tensorboard_log": needs to be set for logging callbacks. Set to newly created temp dir if not provided.

        Args:
            algorithm_parameters (Optional[Dict]): Parameters passed by user (in .__init__ or .load_from_file).

        Returns:
            algorithm_parameters (Dict): Parameter dictionary with filled up default parameter entries

        """
        if not algorithm_parameters:
            algorithm_parameters = {}

        if "policy" not in algorithm_parameters:
            algorithm_parameters.update({"policy": "MlpPolicy"})

        # Existing tensorboard log paths can be used (e.g., for continuing training of downloaded agents).
        # If not provided, tensorboard will be logged to newly created temp dir.
        if "tensorboard_log" not in algorithm_parameters:
            tensorboard_log_path = tempfile.mkdtemp()
            algorithm_parameters.update({"tensorboard_log": tensorboard_log_path})

        return algorithm_parameters

import copy
import inspect
from abc import ABC, abstractmethod
from typing import Any, Optional, Type

import gymnasium as gym
import numpy
import numpy as np
import pettingzoo
import torch.nn
from pettingzoo.utils.wrappers import BaseParallelWrapper
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper

from rl_framework.util.types import Environment


class FeaturesExtractor(ABC, torch.nn.Module):
    preprocessed_observation_space: gym.spaces.Space = None
    output_dim: int = None

    def preprocess(self, observations) -> numpy.ndarray:
        """
        Preprocess observations before feeding them to the model.
        This method by default does nothing (just converting observations into a numpy.ndarray in case they aren't yet),
            but it can be overridden for other preprocessing procedures.

        Args:
            observations: Observations to preprocess.

        Returns:
            observations: Preprocessed observations as numpy array.

        """
        return numpy.asarray(observations)

    @abstractmethod
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


def encode_observations_with_features_extractor(
    observations: list[Any], features_extractor: FeaturesExtractor
) -> np.ndarray:
    features = features_extractor.forward(torch.as_tensor(np.array(observations))).detach().numpy()
    assert len(features) == len(observations)
    return features


def _preprocess_one(features_extractor: "FeaturesExtractor", observation):
    """Preprocess a single observation (`preprocess` takes a batch)."""
    return features_extractor.preprocess(np.array([observation]))[0]


def _preprocessed_space(features_extractor: "FeaturesExtractor", observation_space: gym.spaces.Space):
    if features_extractor.preprocessed_observation_space is not None:
        return features_extractor.preprocessed_observation_space
    return observation_space


class FeaturesExtractorPreprocessingGymWrapper(gym.ObservationWrapper):
    def __init__(self, env, features_extractor: "FeaturesExtractor"):
        super().__init__(env)
        self.features_extractor = features_extractor
        self.observation_space = _preprocessed_space(features_extractor, env.observation_space)

    def observation(self, observation):
        return _preprocess_one(self.features_extractor, observation)


class FeaturesExtractorPreprocessingPettingzooWrapper(BaseParallelWrapper):
    def __init__(self, env, features_extractor: "FeaturesExtractor"):
        super().__init__(env)
        self.features_extractor = features_extractor

    def observation_space(self, agent):
        return _preprocessed_space(self.features_extractor, self.env.observation_space(agent))

    def _preprocess(self, observations: dict) -> dict:
        return {
            agent: _preprocess_one(self.features_extractor, observation) for agent, observation in observations.items()
        }

    def reset(self, seed=None, options=None):
        observations, infos = self.env.reset(seed=seed, options=options)
        return self._preprocess(observations), infos

    def step(self, actions: dict):
        observations, rewards, terminations, truncations, infos = self.env.step(actions)
        # Last observation of a finished episode of an agent which restarted on its own (gymnasium's auto-reset
        #   convention, see StableBaselinesAgent.train); it is preprocessed like the returned observations
        final_observations = {
            agent: info["final_observation"] for agent, info in infos.items() if "final_observation" in info
        }
        for agent, final_observation in self._preprocess(final_observations).items():
            infos[agent]["final_observation"] = final_observation
        return self._preprocess(observations), rewards, terminations, truncations, infos


class FeaturesExtractorPreprocessingVecEnvWrapper(VecEnvWrapper):
    """
    Preprocesses the observations of a VecEnv (one `preprocess` call per batch), including the `terminal_observation`
    infos of done sub-environments. Dict observations (a dict of batched arrays in SB3) are passed to `preprocess`
    like those of a gym.Env: as an array of one dict per sub-environment.
    """

    def __init__(self, venv: VecEnv, features_extractor: "FeaturesExtractor"):
        super().__init__(venv, observation_space=_preprocessed_space(features_extractor, venv.observation_space))
        self.features_extractor = features_extractor

    def _preprocess(self, observations):
        if isinstance(observations, dict):
            batch = np.empty(self.num_envs, dtype=object)
            batch[:] = [{key: value[i] for key, value in observations.items()} for i in range(self.num_envs)]
            observations = batch
        return self.features_extractor.preprocess(observations)

    def reset(self):
        return self._preprocess(self.venv.reset())

    def step_wait(self):
        observations, rewards, dones, infos = self.venv.step_wait()
        for info in infos:
            if "terminal_observation" in info:
                info["terminal_observation"] = _preprocess_one(self.features_extractor, info["terminal_observation"])
        return self._preprocess(observations), rewards, dones, infos


class PreprocessedEnvironmentFunction:
    """
    Environment function of an EnvironmentFactory whose environments are wrapped with the features extractor's
    preprocessing (picklable, e.g., for subprocesses which create their environments).
    """

    def __init__(self, env_func, features_extractor: "FeaturesExtractor"):
        self.env_func = env_func
        self.features_extractor = features_extractor

    def __call__(self):
        environment = self.env_func()
        if isinstance(environment, list):
            return [
                wrap_environment_with_features_extractor_preprocessor(env, self.features_extractor)
                for env in environment
            ]
        return wrap_environment_with_features_extractor_preprocessor(environment, self.features_extractor)


def wrap_environment_with_features_extractor_preprocessor(
    environment: Environment, features_extractor: "FeaturesExtractor"
) -> Environment:
    """
    Wrap an environment so that its observations are preprocessed with the features extractor (`preprocess`).

    - gym.Env and PettingZoo ParallelEnv: every observation (also `final_observation` infos of PettingZoo agents which
      restart on their own).
    - VecEnv: every batch of observations (also `terminal_observation` infos).
    - EnvironmentFactory (stub_environment, env_return_function): the stub environment (for its observation space)
      and the environments the function creates (gym.Env, list of gym.Env or VecEnv), when they are created.
    """
    if isinstance(environment, pettingzoo.ParallelEnv):
        return FeaturesExtractorPreprocessingPettingzooWrapper(environment, features_extractor)
    if isinstance(environment, gym.Env):
        return FeaturesExtractorPreprocessingGymWrapper(environment, features_extractor)
    if isinstance(environment, VecEnv):
        return FeaturesExtractorPreprocessingVecEnvWrapper(environment, features_extractor)
    if isinstance(environment, tuple):
        stub_environment, env_func = environment
        return (
            FeaturesExtractorPreprocessingGymWrapper(stub_environment, features_extractor),
            PreprocessedEnvironmentFunction(env_func, features_extractor),
        )
    raise TypeError(
        f"Environment of type {type(environment).__name__} is not supported for using features_extractor (supported: "
        "gym.Env, pettingzoo.ParallelEnv, VecEnv and EnvironmentFactory)."
    )


def get_sb3_policy_kwargs_for_features_extractor(
    features_extractor: FeaturesExtractor,
    policy_class: Optional[Type[BasePolicy]] = None,
    policy_kwargs: Optional[dict] = None,
) -> dict:
    """
    Build SB3 `policy_kwargs` which make the policy use the given features extractor.

    Args:
        features_extractor: Features extractor to be used by the policy.
        policy_class: SB3 policy class the kwargs are built for. `share_features_extractor` defaults to True only if
            the policy class accepts it (e.g., `DQNPolicy` does not). If None, the policy class is assumed to accept it.
        policy_kwargs: Existing policy kwargs to extend (not modified in place). An explicitly provided
            `share_features_extractor` is kept.

    Returns:
        policy_kwargs (dict): Keyword arguments to be passed to the policy constructor.
    """
    policy_kwargs = {
        **(policy_kwargs or {}),
        "features_extractor_class": StableBaselinesFeaturesExtractor,
        "features_extractor_kwargs": {"features_extractor": features_extractor},
    }
    if policy_class is None or "share_features_extractor" in inspect.signature(policy_class.__init__).parameters:
        policy_kwargs.setdefault("share_features_extractor", True)
    return policy_kwargs


def check_saved_features_extractor(saved_policy_kwargs: dict, features_extractor: Optional[FeaturesExtractor]) -> None:
    """
    Raise if the features extractor of a loaded SB3 model does not match the agent's one: the model keeps its saved
    features extractor network, while the agent preprocesses the observations with its own features extractor.
    Since the saved features extractor is an unpickled copy, it is compared by class (name), `output_dim`,
    `preprocessed_observation_space` and the shapes of its parameters (their values change in training).

    Args:
        saved_policy_kwargs: `policy_kwargs` of the loaded model.
        features_extractor: Features extractor of the agent (None if it has none).
    """
    saved = (saved_policy_kwargs.get("features_extractor_kwargs") or {}).get("features_extractor")
    check_saved_features_extractor_description(
        describe_features_extractor(saved) if saved is not None else None, features_extractor
    )


def describe_features_extractor(features_extractor: FeaturesExtractor) -> dict:
    """
    Comparable description of a features extractor, picklable to be stored next to saved weights: its class (name),
    `output_dim`, `preprocessed_observation_space` and the shapes of its parameters (not their values, which change in
    training).
    """
    return {
        "class": f"{type(features_extractor).__module__}.{type(features_extractor).__qualname__}",
        "output_dim": features_extractor.output_dim,
        "preprocessed_observation_space": features_extractor.preprocessed_observation_space,
        "parameter_shapes": {name: tuple(value.shape) for name, value in features_extractor.state_dict().items()},
    }


def check_saved_features_extractor_description(
    saved_description: Optional[dict], features_extractor: Optional[FeaturesExtractor]
) -> None:
    """
    Raise if the description of a saved features extractor (see `describe_features_extractor`) does not match the
    agent's features extractor.

    Args:
        saved_description: Description of the saved features extractor (None if the saved model has none).
        features_extractor: Features extractor of the agent (None if it has none).
    """
    saved = saved_description
    if saved is None and features_extractor is None:
        return
    if saved is None or features_extractor is None:
        raise ValueError(
            f"The loaded model {'has no' if saved is None else 'has a'} features extractor, but the agent "
            f"{'has one' if saved is None else 'has none'}."
        )
    specified = describe_features_extractor(features_extractor)

    differences = []
    if saved["class"] != specified["class"]:
        differences.append(f"class: stored {saved['class']}, specified {specified['class']}")
    else:
        for attribute in ("output_dim", "preprocessed_observation_space"):
            if saved[attribute] != specified[attribute]:
                differences.append(f"`{attribute}`: stored {saved[attribute]!r}, specified {specified[attribute]!r}")
        saved_shapes, specified_shapes = saved["parameter_shapes"], specified["parameter_shapes"]
        differences += [
            f"shape of `{name}`: stored {saved_shapes.get(name, 'not set')}, "
            f"specified {specified_shapes.get(name, 'not set')}"
            for name in sorted(saved_shapes.keys() | specified_shapes.keys())
            if saved_shapes.get(name) != specified_shapes.get(name)
        ]
    if differences:
        raise ValueError(
            f"The agent's features extractor does not match the one of the loaded model: {'; '.join(differences)}"
        )


class StableBaselinesFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Space, features_extractor: FeaturesExtractor):
        super().__init__(observation_space=observation_space, features_dim=features_extractor.output_dim)
        # Each SB3 construction gets its own copy, so that target networks (DQN, SAC, TD3) do not share their weights
        #   with the online network (otherwise the in-place Polyak update corrupts the shared weights).
        self.features_extractor = copy.deepcopy(features_extractor)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.features_extractor(observations)

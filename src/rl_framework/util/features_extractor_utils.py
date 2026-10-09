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


def wrap_environment_with_features_extractor_preprocessor(
    environment: Environment, features_extractor: FeaturesExtractor
) -> Environment:
    class FeaturesExtractorPreprocessingGymWrapper(gym.ObservationWrapper):
        def __init__(self, env, features_extractor: FeaturesExtractor):
            super().__init__(env)
            self.features_extractor = features_extractor
            self.observation_space = (
                features_extractor.preprocessed_observation_space
                if features_extractor.preprocessed_observation_space is not None
                else env.observation_space
            )

        def observation(self, observation):
            return self.features_extractor.preprocess(np.array([observation]))[0]

    class FeaturesExtractorPreprocessingPettingzooWrapper(BaseParallelWrapper):
        def __init__(self, env, features_extractor: FeaturesExtractor):
            super().__init__(env)
            self.features_extractor = features_extractor

        def observation_space(self, agent):
            if self.features_extractor.preprocessed_observation_space is not None:
                return self.features_extractor.preprocessed_observation_space
            return self.env.observation_space(agent)

        def _preprocess(self, observations: dict) -> dict:
            return {
                agent: self.features_extractor.preprocess(np.array([observation]))[0]
                for agent, observation in observations.items()
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

    if isinstance(environment, pettingzoo.ParallelEnv):
        wrapped_environment = FeaturesExtractorPreprocessingPettingzooWrapper(environment, features_extractor)
    elif isinstance(environment, gym.Env):
        wrapped_environment = FeaturesExtractorPreprocessingGymWrapper(environment, features_extractor)
    else:
        raise TypeError(
            "Environment must be either a gym.Env or pettingzoo.ParallelEnv. "
            "Other types are not supported yet for using features_extractor."
        )
    return wrapped_environment


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
    specified = features_extractor
    if saved is None and specified is None:
        return
    if saved is None or specified is None:
        raise ValueError(
            f"The loaded model {'has no' if saved is None else 'has a'} features extractor, but the agent "
            f"{'has one' if saved is None else 'has none'}."
        )

    def class_name(extractor: FeaturesExtractor) -> str:
        return f"{type(extractor).__module__}.{type(extractor).__qualname__}"

    def parameter_shapes(extractor: FeaturesExtractor) -> dict:
        return {name: tuple(value.shape) for name, value in extractor.state_dict().items()}

    differences = []
    if class_name(saved) != class_name(specified):
        differences.append(f"class: stored {class_name(saved)}, specified {class_name(specified)}")
    else:
        for attribute in ("output_dim", "preprocessed_observation_space"):
            if getattr(saved, attribute) != getattr(specified, attribute):
                differences.append(
                    f"`{attribute}`: stored {getattr(saved, attribute)!r}, specified {getattr(specified, attribute)!r}"
                )
        saved_shapes, specified_shapes = parameter_shapes(saved), parameter_shapes(specified)
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

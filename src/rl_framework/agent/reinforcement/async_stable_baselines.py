from typing import Dict, List, Optional, Type

import stable_baselines3
from async_gym_agents.agents.async_agent import get_injected_agent
from async_gym_agents.envs.multi_env import IndexableMultiEnv
from stable_baselines3.common.base_class import BaseAlgorithm

from rl_framework.agent.reinforcement.stable_baselines import StableBaselinesAgent
from rl_framework.util import Connector, Environment, FeaturesExtractor


class AsyncStableBaselinesAgent(StableBaselinesAgent):
    def __init__(
        self,
        algorithm_class: Type[BaseAlgorithm] = stable_baselines3.PPO,
        algorithm_parameters: Optional[Dict] = None,
        features_extractor: Optional[FeaturesExtractor] = None,
    ):
        self.use_mp = algorithm_parameters.pop("use_mp", True) if algorithm_parameters else True
        super().__init__(
            get_injected_agent(algorithm_class, use_mp=self.use_mp), algorithm_parameters, features_extractor
        )

    def to_vectorized_env(self, env_fns):
        return IndexableMultiEnv(env_fns)

    def train(
        self,
        total_timesteps: int = 100000,
        connector: Optional[Connector] = None,
        training_environments: List[Environment] = None,
        *args,
        **kwargs,
    ):
        # Multiprocessing support when providing a list of tuples:
        # - each tuple does space declaration for the policy creation
        # (stub env) + method returning an environment
        # - expected type: list[tuple[gymnasium.Env, Callable]]
        if isinstance(training_environments[0], tuple):
            stub_envs, environment_return_fns = map(tuple, zip(*training_environments))
            # `_envs` argument of AsyncAgentInjector class is used to create environments delayed (for multiprocessing)

            original_init = self.algorithm_class.__init__

            def wrapped_init(this, *args, **kwargs):
                kwargs.setdefault("envs", environment_return_fns)  # functions for envs creation
                return original_init(this, *args, **kwargs)

            self.algorithm_class.__init__ = wrapped_init

            # use stub as a train environments
            # for creating policy with the right model properties
            training_environments = stub_envs

        super().train(total_timesteps, connector, training_environments, *args, **kwargs)
        # base sb3 algorithm class doesn't have an implementation of the shutdown method,
        # only our custom implementation of it - has it
        if hasattr(self.algorithm, "shutdown"):
            self.algorithm.shutdown()

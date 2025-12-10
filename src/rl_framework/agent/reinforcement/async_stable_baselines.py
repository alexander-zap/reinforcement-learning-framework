from functools import partial
from typing import Dict, List, Optional, Type, Union

import gymnasium
import pettingzoo
import stable_baselines3
from async_gym_agents.agents.async_agent import get_injected_agent
from async_gym_agents.envs.multi_env import IndexableMultiEnv
from stable_baselines3.common.base_class import BaseAlgorithm, VecEnv

from rl_framework.agent.reinforcement.stable_baselines import StableBaselinesAgent
from rl_framework.util import Connector, FeaturesExtractor


class AsyncStableBaselinesAgent(StableBaselinesAgent):
    def __init__(
        self,
        algorithm_class: Type[BaseAlgorithm] = stable_baselines3.PPO,
        algorithm_parameters: Optional[Dict] = None,
        features_extractor: Optional[FeaturesExtractor] = None,
    ):
        super().__init__(get_injected_agent(algorithm_class), algorithm_parameters, features_extractor)

    def to_vectorized_env(self, env_fns):
        return IndexableMultiEnv(env_fns)

    def train(
        self,
        total_timesteps: int = 100000,
        connector: Optional[Connector] = None,
        training_environments: List[Union[gymnasium.Env, pettingzoo.ParallelEnv, VecEnv, tuple]] = None,
        *args,
        **kwargs,
    ):
        # Multiprocessing support when providing a list of tuples:
        # - each tuple does space declaration for the policy creation (dummy env) + method returning an environment
        # - expected type: list[tuple[gymnasium.Env, Callable]]
        if isinstance(training_environments[0], tuple):
            environment_return_functions, training_environments = map(list, zip(*training_environments))
            # `_envs` argument of AsyncAgentInjector class is used to create environments delayed (for multiprocessing)
            self.algorithm_class.__init__ = partial(self.algorithm_class.__init__, _envs=environment_return_functions)

        super().train(total_timesteps, connector, training_environments, *args, **kwargs)
        self.algorithm.shutdown()

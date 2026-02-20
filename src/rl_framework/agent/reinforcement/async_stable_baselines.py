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
        super().__init__(get_injected_agent(algorithm_class), algorithm_parameters, features_extractor)

    def to_vectorized_env(self, env_fns, stub_env=None):
        return IndexableMultiEnv(env_fns, stub_env)

    def train(
        self,
        total_timesteps: int = 100000,
        connector: Optional[Connector] = None,
        training_environments: List[Environment] = None,
        *args,
        **kwargs,
    ):
        super().train(total_timesteps, connector, training_environments, *args, **kwargs)
        # base sb3 algorithm class doesn't have an implementation of the shutdown method,
        # only our custom implementation of it - has it
        if hasattr(self.algorithm, "shutdown"):
            self.algorithm.shutdown()

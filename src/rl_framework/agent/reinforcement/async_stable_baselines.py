from typing import Dict, Optional, Type

import stable_baselines3
from async_gym_agents.agents.async_agent import get_injected_agent
from async_gym_agents.envs.multi_env import IndexableMultiEnv
from stable_baselines3.common.base_class import BaseAlgorithm

from rl_framework.agent.reinforcement.stable_baselines import StableBaselinesAgent
from rl_framework.util import FeaturesExtractor


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

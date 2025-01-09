from abc import ABC, abstractmethod
from typing import List

import gymnasium as gym

from rl_framework.agent.base_agent import Agent
from rl_framework.util import Connector


class RLAgent(Agent, ABC):
    @abstractmethod
    def train(
        self,
        total_timesteps: int,
        connector: Connector,
        training_environments: List[gym.Env],
        *args,
        **kwargs,
    ):
        raise NotImplementedError

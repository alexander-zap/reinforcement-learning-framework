from abc import ABC, abstractmethod
from typing import List, Optional

from rl_framework.agent.base_agent import Agent
from rl_framework.util import Connector, Environment


class RLAgent(Agent, ABC):
    @abstractmethod
    def train(
        self,
        total_timesteps: int,
        connector: Optional[Connector],
        training_environments: List[Environment],
        *args,
        **kwargs,
    ):
        """
        Method starting training for reinforcement learning agents.

        Args:
            total_timesteps: Amount of interaction timesteps to train the agent on.
            connector: Connector for executing callbacks (e.g., logging metrics and saving checkpoints)
                on training time. Calls need to be declared manually in the code.
                If no connector is provided, callbacks will not be executed.
            training_environments: List of gym (or pettingzoo) interface environments on which the agent
                should be trained on.

        """
        raise NotImplementedError

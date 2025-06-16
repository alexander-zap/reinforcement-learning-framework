from abc import ABC, abstractmethod
from typing import Optional

from rl_framework.agent import Agent
from rl_framework.agent.imitation.episode_sequence import EpisodeSequence
from rl_framework.util import Connector


class ILAgent(Agent, ABC):
    @abstractmethod
    def train(
        self,
        total_timesteps: int,
        connector: Optional[Connector],
        episode_sequence: EpisodeSequence,
        *args,
        **kwargs,
    ):
        """
        Method starting training for imitation learning agents.

        Args:
            total_timesteps: Amount of (recorded) timesteps to train the agent on.
            connector: Connector for executing callbacks (e.g., logging metrics and saving checkpoints)
                on training time. Calls need to be declared manually in the code.
                If no connector is provided, callbacks will not be executed.
            episode_sequence: List of episodes on which the agent should be trained on.
                Each episode consists of a sequence, which has the following format:
                [
                    (obs_t0, action_t0, next_obs_t0, reward_t0, terminated_t0, truncated_t0, info_t0),
                    (obs_t1, action_t1, next_obs_t1, reward_t1, terminated_t1, truncated_t1, info_t1),
                    ...
                ]
                Interpretation: Transition from obs to next_obs (next_obs_t == obs_t+1) with action, receiving reward.
                    Additional information returned about transition to next_obs: terminated, truncated and info.
        """
        raise NotImplementedError

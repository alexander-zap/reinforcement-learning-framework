from typing import Dict, List, Optional, Type

import numpy as np
import stable_baselines3
from async_gym_agents.agents.async_agent import get_injected_agent
from async_gym_agents.envs.multi_env import IndexableMultiEnv
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback

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

    def get_callbacks(self, connector: Connector) -> list[BaseCallback]:
        class AsyncSBUtilizationLoggingCallback(BaseCallback):
            """
            A custom callback that logs after every n episodes:
                - buffer utilization
                - buffer emptiness
                - buffer fullness
                - buffer worker fullness wait time
                - discarded episodes
                - main profiler stats
                - worker profiler stats
            """

            def __init__(self, connector, logging_frequency=1000, verbose=0):
                """
                Args:
                    verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
                """
                super().__init__(verbose)
                self.connector = connector
                self.logging_frequency = logging_frequency
                self.shared_episode_counter: int = 0

            def _on_step(self) -> bool:
                done_indices = np.where(self.locals["dones"] == True)[0]
                if done_indices.size != 0:
                    for _ in done_indices:
                        self.shared_episode_counter += 1
                        log_this_episode = self.shared_episode_counter % self.logging_frequency == 0

                        if log_this_episode:
                            report: Dict[str, object] = self.model.get_profiler_report()

                            buffer_utilization = report["buffer"]["utilization"]
                            buffer_emptiness = report["buffer"]["emptiness"]
                            buffer_full_push_fraction = report["buffer"]["full_push_fraction"]
                            buffer_avg_push_time = report["buffer"]["avg_push_time_seconds"]
                            discarded_episodes_fraction = report["buffer"]["discarded_episodes_fraction"]

                            self.connector.log_value_with_timestep(
                                self.num_timesteps,
                                buffer_utilization,
                                value_name="Buffer Utilization",
                                title_name="Buffer Profiler Stats",
                            )
                            self.connector.log_value_with_timestep(
                                self.num_timesteps,
                                buffer_emptiness,
                                value_name="Buffer Emptiness",
                                title_name="Buffer Profiler Stats",
                            )
                            self.connector.log_value_with_timestep(
                                self.num_timesteps,
                                buffer_full_push_fraction,
                                value_name="Buffer Fullness",
                                title_name="Buffer Profiler Stats",
                            )
                            self.connector.log_value_with_timestep(
                                self.num_timesteps,
                                buffer_avg_push_time,
                                value_name="Buffer Worker Fullness Wait Time",
                                title_name="Buffer Profiler Stats",
                            )
                            self.connector.log_value_with_timestep(
                                self.num_timesteps,
                                discarded_episodes_fraction,
                                value_name="Discarded Episodes",
                                title_name="Buffer Profiler Stats",
                            )

                            main_stats = report["main"]

                            for phase in main_stats.keys():
                                for key, value in main_stats[phase].items():
                                    self.connector.log_value_with_timestep(
                                        self.num_timesteps,
                                        value,
                                        value_name=f"{phase}/{key}",
                                        title_name="Main Profiler Stats",
                                    )

                            worker_stats = report["worker"]

                            for phase in worker_stats.keys():
                                for key, value in worker_stats[phase].items():
                                    self.connector.log_value_with_timestep(
                                        self.num_timesteps,
                                        value,
                                        value_name=f"{phase}/{key}",
                                        title_name="Worker Profiler Stats",
                                    )
                return True

        callbacks = super().get_callbacks(connector)
        callback_async_utilization_logging_frequency = self.callback_parameters.get(
            "callback_async_utilization_logging_interval", 1000
        )
        callbacks.append(
            AsyncSBUtilizationLoggingCallback(connector, logging_frequency=callback_async_utilization_logging_frequency)
        )
        return callbacks

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

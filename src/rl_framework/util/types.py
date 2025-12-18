from typing import Callable, List, Union

import gymnasium as gym
from pettingzoo import ParallelEnv
from stable_baselines3.common.vec_env import VecEnv

# tuple of
#   - gymnasium.Env (defining observation and action space)
#   - Callable returning a gymnasium.Env or a list of gym.Env
EnvironmentFactory = tuple[gym.Env, Callable[[], Union[gym.Env, List[gym.Env]]]]

Environment = Union[gym.Env, ParallelEnv, VecEnv, EnvironmentFactory]

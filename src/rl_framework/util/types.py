from typing import Callable, List, Union

import gymnasium as gym
from pettingzoo import ParallelEnv
from stable_baselines3.common.vec_env import VecEnv

# Tuple of
#   - gymnasium.Env (stub env, defining observation and action space)
#   - Callable returning a gymnasium.Env or a list of gym.Env or a VecEnv
# This is the preferred format for using multiprocessing.
# It allows to delay environment creation until the process is created. Otherwise, there will be pickling errors.
EnvironmentFactory = tuple[gym.Env, Callable[[], Union[gym.Env, List[gym.Env], VecEnv]]]

Environment = Union[gym.Env, ParallelEnv, VecEnv, EnvironmentFactory]

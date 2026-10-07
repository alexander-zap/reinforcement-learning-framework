"""Small environments, agents, features extractors and connectors shared by the tests."""

import functools

import gymnasium as gym
import numpy as np
import pettingzoo
import torch as th
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv

from rl_framework.agent import ImitationAgent, StableBaselinesAgent
from rl_framework.agent.base_agent import Agent
from rl_framework.agent.imitation import D3RLPYAgent
from rl_framework.util import DummyConnector, FeaturesExtractor

BOX = spaces.Box(-1, 1, (3,), np.float32)


class Toy(gym.Env):
    """Fixed-length episodes (`episode_length` steps), random observations, reward 1 per step."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, observation_space=BOX, action_space=spaces.Discrete(2), episode_length=5, info=None):
        self.observation_space = observation_space
        self.action_space = action_space
        self.episode_length = episode_length
        self.info = info or {}
        self.render_mode = "rgb_array"
        self.t = 0
        self.closed = False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        return self.observation_space.sample(), {}

    def step(self, action):
        self.t += 1
        return self.observation_space.sample(), 1.0, self.t >= self.episode_length, False, dict(self.info)

    def render(self):
        return np.full((16, 16, 3), self.t, np.uint8)

    def close(self):
        self.closed = True


class ToyParallel(pettingzoo.ParallelEnv):
    """Two agents, observations from BOX, reward 1 per step, all agents terminate after `episode_length` steps."""

    metadata = {"name": "toy_parallel"}

    def __init__(self, episode_length=5):
        self.possible_agents = ["a", "b"]
        self.agents = list(self.possible_agents)
        self.episode_length = episode_length
        self.render_mode = None
        self.t = 0

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return BOX

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return spaces.Discrete(2)

    def reset(self, seed=None, options=None):
        self.agents = list(self.possible_agents)
        self.t = 0
        return {agent: BOX.sample() for agent in self.agents}, {agent: {} for agent in self.agents}

    def step(self, actions):
        self.t += 1
        done = self.t >= self.episode_length
        observations = {agent: BOX.sample() for agent in self.agents}
        rewards = {agent: 1.0 for agent in self.agents}
        terminations = {agent: done for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        if done:
            self.agents = []
        return observations, rewards, terminations, truncations, infos


class Linear(FeaturesExtractor):
    output_dim = 4

    def __init__(self, input_dim=3):
        super().__init__()
        self.layer = th.nn.Linear(input_dim, self.output_dim)

    def forward(self, observations):
        return self.layer(observations.float())


class SB3Agent(StableBaselinesAgent):
    """StableBaselinesAgent which vectorizes in-process (no subprocesses)."""

    def to_vectorized_env(self, env_fns, stub_env=None):
        return DummyVecEnv(env_fns)


class InProcessImitationAgent(ImitationAgent):
    @staticmethod
    def to_vectorized_env(env_fns):
        return DummyVecEnv(env_fns)


class InProcessD3RLPYAgent(D3RLPYAgent):
    @staticmethod
    def to_vectorized_env(env_fns):
        return DummyVecEnv(env_fns)


class ConstantActionAgent(Agent):
    """Always takes action 0; records the `deterministic` flag of each call."""

    algorithm = None

    def __init__(self, features_extractor=None):
        super().__init__(algorithm_class=None, algorithm_parameters=None, features_extractor=features_extractor)
        self.deterministic_flags = set()
        self.observations = []

    def choose_action(self, observation, deterministic=False, *args, **kwargs):
        self.deterministic_flags.add(deterministic)
        self.observations.append(observation)
        return 0

    save_policy_as_onnx = save_to_file = load_from_file = None


class RecordingConnector(DummyConnector):
    """Keeps the base connector's logging and records uploads and downloads."""

    def __init__(self, download_path=None):
        super().__init__()
        self.uploads = []
        self.download_path = download_path

    def upload(self, agent, video_recording_environment=None, checkpoint_id=None, *args, **kwargs):
        self.uploads.append(
            {"agent": agent, "video_recording_environment": video_recording_environment, "checkpoint_id": checkpoint_id}
        )

    def download(self, *args, **kwargs):
        return self.download_path

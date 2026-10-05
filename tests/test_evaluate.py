import gymnasium
import numpy as np
import pettingzoo

from rl_framework.agent.base_agent import Agent

EPISODE_LENGTH = 3
N_AGENTS = 2


class ConstantParallelEnv(pettingzoo.ParallelEnv):
    """Every agent gets a reward of 1 per step; all agents terminate after EPISODE_LENGTH steps."""

    metadata = {"name": "constant_parallel_env"}

    def __init__(self):
        self.possible_agents = [f"agent_{index}" for index in range(N_AGENTS)]
        self.agents = list(self.possible_agents)
        self.steps = 0
        self.closed = False

    def observation_space(self, agent):
        return gymnasium.spaces.Box(low=0.0, high=1.0, shape=(1,))

    def action_space(self, agent):
        return gymnasium.spaces.Discrete(2)

    def reset(self, seed=None, options=None):
        self.agents = list(self.possible_agents)
        self.steps = 0
        return {agent: np.zeros(1) for agent in self.agents}, {agent: {} for agent in self.agents}

    def step(self, actions):
        self.steps += 1
        done = self.steps >= EPISODE_LENGTH
        observations = {agent: np.zeros(1) for agent in self.agents}
        rewards = {agent: 1.0 for agent in self.agents}
        terminations = {agent: done for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        return observations, rewards, terminations, truncations, infos

    def close(self):
        self.closed = True


class ConstantActionAgent(Agent):
    algorithm = None

    def __init__(self):
        super().__init__(algorithm_class=None, algorithm_parameters=None, features_extractor=None)

    def choose_action(self, observation, deterministic, *args, **kwargs):
        return 0

    def save_as_onnx(self, file_path, *args, **kwargs):
        raise NotImplementedError

    def save_to_file(self, file_path, *args, **kwargs):
        raise NotImplementedError

    def load_from_file(self, file_path, algorithm_parameters, *args, **kwargs):
        raise NotImplementedError


def test_evaluate_on_pettingzoo_parallel_environments():
    environments = [ConstantParallelEnv(), ConstantParallelEnv()]

    mean_reward, std_reward = ConstantActionAgent().evaluate(evaluation_environments=environments, n_eval_episodes=4)

    assert mean_reward == EPISODE_LENGTH
    assert std_reward == 0.0
    assert all(environment.closed for environment in environments)

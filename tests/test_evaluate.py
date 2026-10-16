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

    def save_policy_as_onnx(self, file_path, *args, **kwargs):
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


class ContextRecordingAgent(ConstantActionAgent):
    """Records the episode context which `evaluate` passes to `choose_action`."""

    def __init__(self):
        super().__init__()
        self.calls = []

    def choose_action(self, observation, deterministic, *args, **kwargs):
        self.calls.append((kwargs["episode_key"], kwargs["reward"], kwargs["episode_start"]))
        return 0

    def contexts_by_key(self):
        contexts = {}
        for episode_key, reward, episode_start in self.calls:
            contexts.setdefault(episode_key, []).append((reward, episode_start))
        return contexts


def test_evaluate_passes_the_episode_context_of_every_pettingzoo_agent():
    agent = ContextRecordingAgent()

    agent.evaluate(evaluation_environments=[ConstantParallelEnv(), ConstantParallelEnv()], n_eval_episodes=4)

    contexts = agent.contexts_by_key()
    # Two agents in each of the two environments
    assert len(contexts) == 4
    assert {episode_key[1] for episode_key in contexts} == {"agent_0", "agent_1"}
    for agent_contexts in contexts.values():
        # Reset, two steps with reward 1, the done step (a start: the agent is still active) and the reset
        assert agent_contexts == [(0.0, True), (1.0, False), (1.0, False), (0.0, True), (0.0, True)]


class StepRewardEnv(gymnasium.Env):
    """Episodes of EPISODE_LENGTH steps; the reward of a step is its number (1, 2, ...)."""

    observation_space = gymnasium.spaces.Box(-1, 1, (1,), np.float32)
    action_space = gymnasium.spaces.Discrete(2)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return np.zeros(1, np.float32), {}

    def step(self, action):
        self.steps += 1
        return np.zeros(1, np.float32), float(self.steps), self.steps >= EPISODE_LENGTH, False, {}


def test_evaluate_passes_the_episode_context_of_every_sub_environment():
    agent = ContextRecordingAgent()

    agent.evaluate(evaluation_environments=[StepRewardEnv(), StepRewardEnv()], n_eval_episodes=2)

    contexts = agent.contexts_by_key()
    assert sorted(episode_key[1] for episode_key in contexts) == [0, 1]
    for sub_environment_contexts in contexts.values():
        # Reset, steps with the previous step's reward, the next episode (automatic reset) without reward
        assert sub_environment_contexts == [(0.0, True), (1.0, False), (2.0, False), (0.0, True)]

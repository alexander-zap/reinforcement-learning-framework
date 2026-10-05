"""Agent.evaluate on gym/VecEnv environments: episode quota and evaluation metric logging.

Strict xfails document unresolved findings; they assert the desired behavior.
"""

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv

from rl_framework.agent.base_agent import Agent
from rl_framework.util import DummyConnector


class CountingEnv(gym.Env):
    """One-step episodes; the k-th episode (from 0) has reward k and step metric k."""

    observation_space = spaces.Box(-1, 1, (1,), np.float32)
    action_space = spaces.Discrete(2)

    def __init__(self):
        self.episodes = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(1, np.float32), {}

    def step(self, action):
        value = float(self.episodes)
        self.episodes += 1
        return np.zeros(1, np.float32), value, True, False, {"step_metric_x": value}


class ConstantActionAgent(Agent):
    algorithm = None

    def __init__(self):
        super().__init__(algorithm_class=None, algorithm_parameters=None, features_extractor=None)

    def choose_action(self, observation, deterministic, *args, **kwargs):
        return 0

    save_as_onnx = save_to_file = load_from_file = None


class RecordingConnector(DummyConnector):
    def __init__(self):
        super().__init__()
        self.scalars = []

    def log_value_with_timestep(self, timestep, value_scalar, value_name, title_name=None):
        self.scalars.append((title_name or value_name, value_name, timestep, float(value_scalar)))


def evaluate(environments, n_eval_episodes, connector=None):
    return ConstantActionAgent().evaluate(
        evaluation_environments=environments,
        n_eval_episodes=n_eval_episodes,
        connector=connector or DummyConnector(),
        logging_frequency=1,
    )


@pytest.mark.xfail(
    strict=True,
    reason="base_agent.py:375 gives each env n_eval_episodes // len(envs) + 1 episodes and all count in mean/std",
)
def test_evaluate_averages_exactly_the_requested_episodes():
    mean_reward, _ = evaluate([CountingEnv()], n_eval_episodes=4)

    assert mean_reward == np.mean([0, 1, 2, 3])


@pytest.mark.xfail(
    strict=True,
    reason="base_agent.py:358-362 every env thread logs to the same series at step len(local_episode_rewards)",
)
def test_evaluation_metrics_of_different_environments_do_not_collide():
    connector = RecordingConnector()
    evaluate([DummyVecEnv([CountingEnv]), DummyVecEnv([CountingEnv])], n_eval_episodes=4, connector=connector)

    points = [(title, series, step) for title, series, step, _ in connector.scalars]
    assert points
    assert len(points) == len(set(points))


@pytest.mark.xfail(
    strict=True,
    reason="base_agent.py:358 never calls reset_multi_episode_trackers, so each logged mean covers all past episodes",
)
def test_evaluation_step_metric_means_cover_only_the_logging_window():
    connector = RecordingConnector()
    evaluate([CountingEnv()], n_eval_episodes=3, connector=connector)

    means = [value for _, series, _, value in connector.scalars if series == "Mean - x"]
    assert means == list(range(len(means)))
    assert len(means) >= 3

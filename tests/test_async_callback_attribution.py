"""Per-agent attribution of LoggingCallback/ResetInfoCallback in AsyncSB3 episode-batched mode.

Contract: episodes collected by different async workers are attributed to distinct agent indices.

Strict xfails document unresolved findings; they assert the desired behavior.
"""

import re

import gymnasium as gym
import numpy as np
import pytest
import stable_baselines3
from gymnasium import spaces

from rl_framework.agent.reinforcement.async_stable_baselines import (
    AsyncStableBaselinesAgent,
)
from rl_framework.util import DummyConnector, MetricAggregator

N_WORKERS = 2


class Toy(gym.Env):
    observation_space = spaces.Box(-1, 1, (4,), np.float32)
    action_space = spaces.Discrete(3)

    def __init__(self, worker):
        self.worker = worker

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(4, np.float32), {"worker": self.worker}

    def step(self, action):
        return np.zeros(4, np.float32), 1.0, True, False, {}


class RecordingConnector(DummyConnector):
    def __init__(self):
        super().__init__()
        self.dict_names = []
        self.workers_seen = set()

    def log_dict(self, dict_to_log, dict_name):
        self.dict_names.append(dict_name)
        self.workers_seen.add(dict_to_log.get("worker"))


@pytest.fixture(scope="module")
def trained():
    """Train with two async workers, recording logged reset infos and aggregated-metric agent indices."""
    logged_agent_indices = []
    original_log = MetricAggregator.log_aggregated_metrics

    def record_agent_index(self, agent_index, *args, **kwargs):
        logged_agent_indices.append(int(agent_index))
        return original_log(self, agent_index, *args, **kwargs)

    mp = pytest.MonkeyPatch()
    mp.setattr(MetricAggregator, "log_aggregated_metrics", record_agent_index)
    connector = RecordingConnector()
    agent = AsyncStableBaselinesAgent(
        algorithm_class=stable_baselines3.PPO,
        algorithm_parameters={
            "device": "cpu",
            "seed": 0,
            "n_steps": 8,
            "batch_size": 4,
            "n_epochs": 1,
            "tensorboard_log": None,
            "policy_kwargs": {"net_arch": [8]},
            "use_mp": False,
            "worker_join_timeout": 5.0,
        },
    )
    try:
        agent.train(
            total_timesteps=32,
            connector=connector,
            training_environments=[Toy(worker) for worker in range(N_WORKERS)],
        )
    finally:
        mp.undo()
    return connector, logged_agent_indices


def reset_info_agent_indices(connector):
    return {int(m.group(1)) for name in connector.dict_names if (m := re.match(r"Reset Info - Agent (\d+)", name))}


@pytest.mark.xfail(
    strict=True,
    reason="sb3_training_callbacks.py:108 every async episode batch has n_envs=1, so done_index is always 0",
)
def test_logging_callback_attributes_workers_to_distinct_agents(trained):
    connector, logged_agent_indices = trained

    assert connector.workers_seen == set(range(N_WORKERS))  # both workers' episodes reached the callbacks
    assert set(logged_agent_indices) == set(range(N_WORKERS))


@pytest.mark.xfail(
    strict=True,
    reason="sb3_training_callbacks.py:367-374 every async episode batch has n_envs=1, so agent index is always 0",
)
def test_reset_info_callback_attributes_workers_to_distinct_agents(trained):
    connector, _ = trained

    assert connector.workers_seen == set(range(N_WORKERS))  # both workers' episodes reached the callbacks
    assert reset_info_agent_indices(connector) == set(range(N_WORKERS))

"""Attribution of LoggingCallback/ResetInfoCallback in AsyncSB3 episode-batched mode.

Contract: ResetInfoCallback attributes episodes collected by different async workers to distinct agent indices
(async-gym-agents passes each episode's origin, its worker and sub-environment, since 0.5.0), while LoggingCallback
aggregates the episodes of all workers together.
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
    """Train with two async workers, recording logged reset infos and the aggregators which logged metrics."""
    logging_aggregators = set()
    original_log = MetricAggregator.log_and_clear_aggregated_metrics

    def record_aggregator(self, *args, **kwargs):
        logging_aggregators.add(id(self))
        return original_log(self, *args, **kwargs)

    mp = pytest.MonkeyPatch()
    mp.setattr(MetricAggregator, "log_and_clear_aggregated_metrics", record_aggregator)
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
    return connector, logging_aggregators


def reset_info_agent_indices(connector):
    return {int(m.group(1)) for name in connector.dict_names if (m := re.match(r"Reset Info - Agent (\d+)", name))}


def test_logging_callback_aggregates_all_workers_together(trained):
    connector, logging_aggregators = trained

    assert connector.workers_seen == set(range(N_WORKERS))  # both workers' episodes reached the callbacks
    assert len(logging_aggregators) == 1  # all workers' metrics are aggregated and logged together


def test_reset_info_callback_attributes_workers_to_distinct_agents(trained):
    connector, _ = trained

    assert connector.workers_seen == set(range(N_WORKERS))  # both workers' episodes reached the callbacks
    assert reset_info_agent_indices(connector) == set(range(N_WORKERS))

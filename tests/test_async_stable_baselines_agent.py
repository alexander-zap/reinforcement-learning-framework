"""AsyncStableBaselinesAgent: injected algorithm, vectorization, utilization logging callback and worker shutdown.

Strict xfails document unresolved findings; they assert the desired behavior.
"""

from unittest.mock import Mock

import numpy as np
import pytest
from async_gym_agents.envs.multi_env import IndexableMultiEnv
from stable_baselines3 import PPO

from rl_framework.agent import AsyncStableBaselinesAgent, StableBaselinesAgent
from tests.toys import RecordingConnector


def make_agent(**callback_kwargs):
    return AsyncStableBaselinesAgent(PPO, {"device": "cpu", "callback_kwargs": callback_kwargs})


def test_algorithm_class_is_an_injected_subclass_of_the_given_algorithm():
    agent = make_agent()
    assert issubclass(agent.algorithm_class, PPO)
    assert agent.algorithm_class is not PPO


def test_vectorization_uses_indexable_multi_env(monkeypatch):
    created = []
    monkeypatch.setattr(IndexableMultiEnv, "__init__", lambda self, *args: created.append(args))

    venv = make_agent().to_vectorized_env(["env_fn"], stub_env="stub")

    assert isinstance(venv, IndexableMultiEnv)
    assert created == [(["env_fn"], "stub")]


def profiler_report():
    return {
        "main": {"collect": {"mean": 1.0}},
        "worker": {"step": {"mean": 2.0}},
        "buffer": {"utilization": 0.5, "unknown": None},
        "worker_sync": {},
        "transport": {},
        "assembly": {},
        "policy": {"lag": 3},
    }


def utilization_callback(connector, **callback_kwargs):
    callback = make_agent(**callback_kwargs).get_callbacks(connector)[-1]
    callback.model = Mock(get_profiler_report=Mock(return_value=profiler_report()))
    return callback


def test_utilization_callback_is_added_with_configured_interval():
    callback = utilization_callback(RecordingConnector(), callback_async_utilization_logging_interval=3)
    assert type(callback).__name__ == "AsyncSBUtilizationLoggingCallback"
    assert callback.logging_frequency == 3


def test_utilization_callback_logs_profiler_report_every_n_episodes():
    connector = RecordingConnector()
    callback = utilization_callback(connector, callback_async_utilization_logging_interval=2)
    callback.num_timesteps = 10

    callback.locals = {"dones": np.array([True, False])}
    assert callback._on_step() is True
    assert connector.value_sequences_to_log == {}

    callback.locals = {"dones": np.array([True, True])}
    callback._on_step()

    assert connector.value_sequences_to_log == {
        "collect": [(10, 1.0)],
        "step": [(10, 2.0)],
        "utilization": [(10, 0.5)],
        "lag": [(10, 3)],
    }


def test_train_shuts_down_workers_after_training(monkeypatch):
    monkeypatch.setattr(StableBaselinesAgent, "train", lambda self, *args, **kwargs: None)
    agent = make_agent()
    agent.algorithm = Mock()

    agent.train(total_timesteps=1, training_environments=["env"])

    agent.algorithm.shutdown.assert_called_once()


def test_train_shuts_down_workers_when_training_fails(monkeypatch):
    def failing_train(self, *args, **kwargs):
        raise RuntimeError("worker failed")

    monkeypatch.setattr(StableBaselinesAgent, "train", failing_train)
    agent = make_agent()
    agent.algorithm = Mock()

    with pytest.raises(RuntimeError):
        agent.train(total_timesteps=1, training_environments=["env"])

    agent.algorithm.shutdown.assert_called_once()

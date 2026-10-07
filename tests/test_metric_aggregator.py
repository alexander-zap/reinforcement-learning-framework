"""MetricAggregator episode bookkeeping with several environments stepping together.

Strict xfails document unresolved findings; they assert the desired behavior.
"""

import numpy as np
import pytest

from rl_framework.util import DummyConnector, MetricAggregator


def finish_one_episode(dones, infos):
    aggregator = MetricAggregator(connector=DummyConnector())
    aggregator.aggregate_step(np.zeros((2, 1)), [0, 0], np.ones(2), np.array(dones), infos)
    return aggregator.episode_rewards


def test_discarded_episode_of_last_env_is_not_recorded():
    assert finish_one_episode([False, True], [{}, {"discard": True}]) == {}


@pytest.mark.xfail(
    strict=True,
    reason="metric_logging_utils.py:66 checks infos[agent_index] "
    "(stale loop variable = last env) not infos[done_index]",
)
def test_discarded_episode_of_first_env_is_not_recorded():
    assert finish_one_episode([True, False], [{"discard": True}, {}]) == {}

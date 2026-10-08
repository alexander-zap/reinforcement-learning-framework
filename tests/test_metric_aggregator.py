"""MetricAggregator episode bookkeeping with several environments stepping together.

Strict xfails document unresolved findings; they assert the desired behavior.
"""

import numpy as np

from rl_framework.util import DummyConnector, MetricAggregator


def finish_one_episode(dones, infos):
    aggregator = MetricAggregator(connector=DummyConnector())
    aggregator.aggregate_step(np.zeros((2, 1)), [0, 0], np.ones(2), np.array(dones), infos)
    return aggregator.episode_rewards


def test_discarded_episode_of_last_env_is_not_recorded():
    assert finish_one_episode([False, True], [{}, {"discard": True}]) == {}


def test_discarded_episode_of_first_env_is_not_recorded():
    assert finish_one_episode([True, False], [{"discard": True}, {}]) == {}


def test_discarded_episode_reward_does_not_leak_into_next_episode():
    aggregator = MetricAggregator(connector=DummyConnector())
    aggregator.aggregate_step(None, [0], np.array([5.0]), np.array([True]), [{"discard": True}])
    aggregator.aggregate_step(None, [0], np.array([1.0]), np.array([True]), [{}])

    assert aggregator.episode_rewards == {0: [1.0]}


class LoggedValues(DummyConnector):
    def __init__(self):
        super().__init__()
        self.scalars = {}
        self.histograms = {}

    def log_value_with_timestep(self, timestep, value_scalar, value_name, title_name=None):
        self.scalars[(title_name, value_name)] = (timestep, float(value_scalar))

    def log_histogram_with_timestep(self, timestep, histogram_values, histogram_name):
        self.histograms[histogram_name] = (timestep, np.asarray(histogram_values).tolist())


def step(aggregator, rewards, dones, infos=None, actions=None):
    n = len(rewards)
    aggregator.aggregate_step(
        np.zeros((n, 1)),
        actions if actions is not None else [0] * n,
        np.array(rewards, dtype=float),
        np.array(dones),
        infos or [{} for _ in range(n)],
    )


def test_episode_rewards_accumulate_per_env_and_reset_when_done():
    aggregator = MetricAggregator(connector=DummyConnector())
    step(aggregator, [1, 10], [False, False])
    step(aggregator, [2, 20], [True, False])
    step(aggregator, [3, 30], [True, True])

    assert aggregator.episode_rewards == {0: [3.0, 3.0], 1: [60.0]}
    np.testing.assert_array_equal(aggregator.episode_reward, [0, 0])


def test_step_metrics_are_collected_per_metric_and_env():
    aggregator = MetricAggregator(connector=DummyConnector())
    step(aggregator, [0, 0], [False, False], [{"step_metric_speed": 1, "other": 5}, {}])
    step(aggregator, [0, 0], [False, False], [{"step_metric_speed": 2}, {"step_metric_speed": 7}])

    assert aggregator.episode_step_metrics == {"speed": [[1.0, 2.0], [7.0]]}


def test_episode_end_reasons_are_recorded_for_done_envs_only_and_capped_at_100():
    aggregator = MetricAggregator(connector=DummyConnector())
    step(aggregator, [0, 0], [False, False], [{"episode_end_reason": "ignored"}, {}])
    for _ in range(101):
        step(aggregator, [0, 0], [True, False], [{"episode_end_reason": "won"}, {"episode_end_reason": "ignored"}])

    assert list(aggregator.episode_end_reasons) == [0]
    assert len(aggregator.episode_end_reasons[0]) == 100


def test_actions_are_only_aggregated_when_distributions_are_requested():
    without = MetricAggregator(connector=DummyConnector())
    step(without, [0, 0], [False, False], actions=[1, 2])
    assert without.episode_actions is None

    with_distributions = MetricAggregator(connector=DummyConnector(), aggregate_distributions=True)
    step(with_distributions, [0, 0], [False, False], actions=[1, 2])
    step(with_distributions, [0, 0], [False, False], actions=[3, 4])
    assert with_distributions.episode_actions == [[1, 3], [2, 4]]


def test_log_aggregated_metrics_logs_reward_step_metric_statistics_and_end_reasons():
    connector = LoggedValues()
    aggregator = MetricAggregator(connector=connector)
    step(aggregator, [1], [False], [{"step_metric_speed": 1}])
    step(aggregator, [1], [True], [{"step_metric_speed": 3, "episode_end_reason": "won"}])
    step(aggregator, [4], [True], [{"episode_end_reason": "lost"}])

    aggregator.log_aggregated_metrics(agent_index=0, num_timesteps=42, metric_name_prefix="Eval - ")

    assert connector.scalars == {
        (None, "Eval - Episode reward"): (42, 3.0),
        ("Eval - speed", "Mean - speed"): (42, 2.0),
        ("Eval - speed", "Std - speed"): (42, 1.0),
        ("Eval - speed", "Max - speed"): (42, 3.0),
        ("Eval - speed", "Min - speed"): (42, 1.0),
        ("Eval - Episode end reasons", "Episode end reason - won"): (42, 0.5),
        ("Eval - Episode end reasons", "Episode end reason - lost"): (42, 0.5),
    }
    assert connector.histograms == {}


def test_log_distributions_logs_scalar_actions_and_step_metrics_as_histograms():
    connector = LoggedValues()
    aggregator = MetricAggregator(connector=connector, aggregate_distributions=True)
    step(aggregator, [0], [False], [{"step_metric_speed": 1}], actions=[2])
    step(aggregator, [0], [True], [{"step_metric_speed": 3}], actions=[1])

    aggregator.log_aggregated_metrics(agent_index=0, num_timesteps=5, log_distributions=True)

    assert connector.histograms == {"Action distribution": (5, [2, 1]), "Distribution - speed": (5, [1.0, 3.0])}


def test_log_distributions_logs_one_histogram_per_action_dimension():
    connector = LoggedValues()
    aggregator = MetricAggregator(connector=connector, aggregate_distributions=True)
    step(aggregator, [0], [False], actions=[np.array([1, 5])])
    step(aggregator, [0], [True], actions=[np.array([2, 6])])

    aggregator.log_aggregated_metrics(agent_index=0, num_timesteps=5, log_distributions=True)

    assert connector.histograms == {
        "Action distribution - action dim 0": (5, [1, 2]),
        "Action distribution - action dim 1": (5, [5, 6]),
    }


def test_reset_multi_episode_trackers_clears_only_the_given_env():
    aggregator = MetricAggregator(connector=DummyConnector(), aggregate_distributions=True)
    step(aggregator, [1, 1], [True, True], [{"step_metric_x": 1}, {"step_metric_x": 2}], actions=[0, 1])

    aggregator.reset_multi_episode_trackers(0)

    assert aggregator.episode_rewards == {0: [], 1: [1.0]}
    assert aggregator.episode_step_metrics == {"x": [[], [2.0]]}
    assert aggregator.episode_actions == [[], [1]]

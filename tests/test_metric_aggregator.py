"""MetricAggregator episode bookkeeping with several environments stepping together."""

import numpy as np

from rl_framework.util import DummyConnector, MetricAggregator


def finish_one_episode(dones, infos):
    aggregator = MetricAggregator(connector=DummyConnector())
    aggregator.aggregate_step(np.zeros((2, 1)), [0, 0], np.ones(2), np.array(dones), infos)
    return aggregator.episode_rewards


def test_discarded_episode_of_last_env_is_not_recorded():
    assert finish_one_episode([False, True], [{}, {"discard": True}]) == []


def test_discarded_episode_of_first_env_is_not_recorded():
    assert finish_one_episode([True, False], [{"discard": True}, {}]) == []


def test_discarded_episode_does_not_leak_into_next_episode():
    aggregator = MetricAggregator(connector=DummyConnector())
    aggregator.aggregate_step(None, [0], np.array([5.0]), np.array([True]), [{"discard": True, "step_metric_x": 9}])
    aggregator.aggregate_step(None, [0], np.array([1.0]), np.array([True]), [{}])

    assert aggregator.episode_rewards == [1.0]
    assert aggregator.step_metric_statistics == {}


class LoggedValues(DummyConnector):
    def __init__(self):
        super().__init__()
        self.scalars = {}
        self.histograms = {}

    def log_value_with_timestep(self, timestep, value_scalar, value_name, title_name=None):
        self.scalars[(title_name, value_name)] = (timestep, float(value_scalar))

    def log_histogram_with_timestep(self, timestep, histogram_values, histogram_name):
        self.histograms[histogram_name] = (timestep, np.asarray(histogram_values).tolist())


def step(aggregator, rewards, dones, infos=None, actions=None, index_offset=0):
    n = len(rewards)
    aggregator.aggregate_step(
        np.zeros((n, 1)),
        actions if actions is not None else [0] * n,
        np.array(rewards, dtype=float),
        np.array(dones),
        infos or [{} for _ in range(n)],
        index_offset=index_offset,
    )


def test_episode_rewards_accumulate_per_env_and_are_pooled_when_done():
    aggregator = MetricAggregator(connector=DummyConnector())
    step(aggregator, [1, 10], [False, False])
    step(aggregator, [2, 20], [True, False])
    step(aggregator, [3, 30], [True, True])

    assert aggregator.episode_rewards == [3.0, 3.0, 60.0]
    assert aggregator.running_episode_reward == {}


def test_index_offset_keeps_episodes_of_different_environments_apart():
    aggregator = MetricAggregator(connector=DummyConnector())
    step(aggregator, [1], [False])
    step(aggregator, [10], [True], index_offset=1)
    step(aggregator, [2], [True])

    assert aggregator.episode_rewards == [10.0, 3.0]


def test_step_metrics_of_running_episodes_are_collected_per_env():
    aggregator = MetricAggregator(connector=DummyConnector())
    step(aggregator, [0, 0], [False, False], [{"step_metric_speed": 1, "other": 5}, {}])
    step(aggregator, [0, 0], [False, False], [{"step_metric_speed": 2}, {"step_metric_speed": 7}])

    assert aggregator.running_episode_step_metrics == {0: {"speed": [1.0, 2.0]}, 1: {"speed": [7.0]}}


def test_episode_end_reasons_are_counted_for_done_envs_only_and_cleared_by_a_log():
    aggregator = MetricAggregator(connector=DummyConnector())
    step(aggregator, [0, 0], [False, False], [{"episode_end_reason": "ignored"}, {}])
    for _ in range(3):
        step(aggregator, [0, 0], [True, False], [{"episode_end_reason": "won"}, {"episode_end_reason": "ignored"}])

    assert aggregator.episode_end_reasons == {"won": 3}

    aggregator.log_and_clear_aggregated_metrics(num_timesteps=0)
    assert aggregator.episode_end_reasons == {}
    assert aggregator.seen_end_reasons == {"won"}


def test_actions_are_only_aggregated_when_distributions_are_requested():
    without = MetricAggregator(connector=DummyConnector())
    step(without, [0, 0], [False, False], actions=[1, 2])
    assert without.running_episode_actions == {}

    with_distributions = MetricAggregator(connector=DummyConnector(), aggregate_distributions=True)
    step(with_distributions, [0, 0], [False, False], actions=[1, 2])
    step(with_distributions, [0, 0], [False, False], actions=[3, 4])
    assert with_distributions.running_episode_actions == {0: [1, 3], 1: [2, 4]}


def test_log_logs_reward_step_metric_statistics_and_end_reasons():
    connector = LoggedValues()
    aggregator = MetricAggregator(connector=connector)
    step(aggregator, [1], [False], [{"step_metric_speed": 1}])
    step(aggregator, [1], [True], [{"step_metric_speed": 3, "episode_end_reason": "won"}])
    step(aggregator, [4], [True], [{"episode_end_reason": "lost"}])

    aggregator.log_and_clear_aggregated_metrics(num_timesteps=42, metric_name_prefix="Eval - ")

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

    aggregator.log_and_clear_aggregated_metrics(num_timesteps=5, log_distributions=True)

    assert connector.histograms == {"Action distribution": (5, [2, 1]), "Distribution - speed": (5, [1.0, 3.0])}


def test_log_distributions_logs_one_histogram_per_action_dimension():
    connector = LoggedValues()
    aggregator = MetricAggregator(connector=connector, aggregate_distributions=True)
    step(aggregator, [0], [False], actions=[np.array([1, 5])])
    step(aggregator, [0], [True], actions=[np.array([2, 6])])

    aggregator.log_and_clear_aggregated_metrics(num_timesteps=5, log_distributions=True)

    assert connector.histograms == {
        "Action distribution - action dim 0": (5, [1, 2]),
        "Action distribution - action dim 1": (5, [5, 6]),
    }


def test_log_clears_finished_episodes_only():
    aggregator = MetricAggregator(connector=DummyConnector(), aggregate_distributions=True)
    step(aggregator, [1, 1], [True, False], [{"step_metric_x": 1}, {"step_metric_x": 2}], actions=[0, 1])

    aggregator.log_and_clear_aggregated_metrics(num_timesteps=0)

    assert aggregator.episode_rewards == []
    assert aggregator.step_metric_statistics == {}
    assert aggregator.distribution_step_metric_values == {}
    assert aggregator.distribution_actions == []
    assert aggregator.running_episode_reward == {1: 1.0}
    assert aggregator.running_episode_step_metrics == {1: {"x": [2.0]}}
    assert aggregator.running_episode_actions == {1: [1]}


def test_step_metric_statistics_over_several_episodes_match_numpy_over_all_values():
    connector = LoggedValues()
    aggregator = MetricAggregator(connector=connector)
    values = [[1.0, 4.0, 2.5], [10.0], [-3.0, 7.0]]
    for episode in values:
        for index, value in enumerate(episode):
            step(aggregator, [0], [index == len(episode) - 1], [{"step_metric_x": value}])

    aggregator.log_and_clear_aggregated_metrics(num_timesteps=1)

    all_values = np.concatenate(values)
    for statistic, expected in [("Mean", np.mean), ("Std", np.std), ("Max", np.max), ("Min", np.min)]:
        np.testing.assert_allclose(connector.scalars[("x", f"{statistic} - x")][1], expected(all_values))


def test_values_of_finished_episodes_are_only_kept_for_distributions():
    aggregator = MetricAggregator(connector=DummyConnector())
    step(aggregator, [0, 0], [True, False], [{"step_metric_x": 1}, {"step_metric_x": 2}])

    assert aggregator.running_episode_step_metrics == {1: {"x": [2.0]}}
    assert aggregator.distribution_step_metric_values == {}


def test_running_episodes_are_left_for_the_next_log():
    connector = LoggedValues()
    aggregator = MetricAggregator(connector=connector, aggregate_distributions=True)
    step(aggregator, [1], [True], [{"step_metric_x": 1}], actions=[1])
    step(aggregator, [2], [False], [{"step_metric_x": 3}], actions=[2])

    aggregator.log_and_clear_aggregated_metrics(num_timesteps=1, log_distributions=True)
    assert connector.scalars[("x", "Mean - x")] == (1, 1.0)
    assert connector.histograms["Action distribution"] == (1, [1])

    step(aggregator, [3], [True], [{"step_metric_x": 5}], actions=[3])
    aggregator.log_and_clear_aggregated_metrics(num_timesteps=2, log_distributions=True)
    assert connector.scalars[(None, "Episode reward")] == (2, 5.0)
    assert connector.scalars[("x", "Mean - x")] == (2, 4.0)
    assert connector.histograms["Action distribution"] == (2, [2, 3])


def test_end_reasons_absent_since_the_last_log_are_logged_as_zero():
    connector = LoggedValues()
    aggregator = MetricAggregator(connector=connector)
    step(aggregator, [0], [True], [{"episode_end_reason": "lost"}])
    aggregator.log_and_clear_aggregated_metrics(num_timesteps=0)
    step(aggregator, [0], [True], [{"episode_end_reason": "won"}])

    aggregator.log_and_clear_aggregated_metrics(num_timesteps=1)

    assert connector.scalars[("Episode end reasons", "Episode end reason - won")] == (1, 1.0)
    assert connector.scalars[("Episode end reasons", "Episode end reason - lost")] == (1, 0.0)

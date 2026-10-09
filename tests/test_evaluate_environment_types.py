"""Agent.evaluate on every supported environment type, with features extractor preprocessing and metric logging."""

import numpy as np
import pettingzoo
import pettingzoo.utils
import pytest
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv

from rl_framework.util import FeaturesExtractor
from tests.toys import (
    ConstantActionAgent,
    RecordingConnector,
    RestartingParallel,
    Toy,
    ToyParallel,
)

ONES = spaces.Box(1, 1, (3,), np.float32)


class Doubling(FeaturesExtractor):
    output_dim = 3

    def preprocess(self, observations):
        return np.asarray(observations) * 2

    def forward(self, observations):
        return observations


def evaluate(environments, agent=None, **kwargs):
    agent = agent or ConstantActionAgent()
    return agent.evaluate(evaluation_environments=environments, n_eval_episodes=4, **kwargs)


def test_gym_environments_are_evaluated_together_and_closed():
    environments = [Toy(), Toy(episode_length=3)]

    mean_reward, std_reward = evaluate(environments)

    assert 3.0 < mean_reward < 5.0
    assert std_reward > 0
    assert all(environment.closed for environment in environments)


def test_vec_envs_are_evaluated_in_their_own_threads():
    mean_reward, std_reward = evaluate([DummyVecEnv([Toy, Toy]), DummyVecEnv([Toy])])
    assert (mean_reward, std_reward) == (5.0, 0.0)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: Toy(),
        lambda: [Toy(), Toy()],
        lambda: DummyVecEnv([Toy]),
    ],
    ids=["gym-env", "list-of-gym-envs", "vec-env"],
)
def test_environment_factories_are_instantiated(factory):
    mean_reward, std_reward = evaluate([(Toy(), factory), (Toy(), factory)])
    assert (mean_reward, std_reward) == (5.0, 0.0)


def test_pettingzoo_environments_count_each_agent_episode():
    assert evaluate([ToyParallel()]) == (5.0, 0.0)


class EpisodeRecorder(pettingzoo.utils.BaseParallelWrapper):
    """Records the return of every agent episode the wrapped environment completes."""

    def __init__(self, env):
        super().__init__(env)
        self.returns, self.running = [], {}

    def reset(self, seed=None, options=None):
        self.running = {}
        return self.env.reset(seed=seed, options=options)

    def step(self, actions):
        observations, rewards, terminations, truncations, infos = self.env.step(actions)
        for agent, reward in rewards.items():
            self.running[agent] = self.running.get(agent, 0.0) + reward
            if terminations[agent] or truncations[agent]:
                self.returns.append(self.running.pop(agent))
        return observations, rewards, terminations, truncations, infos


@pytest.mark.parametrize(
    "lengths",
    [
        pytest.param((3, 4), id="restarting-agents"),
        pytest.param((1, 4), id="restarting-agents-with-one-step-episodes"),
    ],
)
def test_pettingzoo_self_restarting_agents_count_every_episode(lengths):
    environment = EpisodeRecorder(RestartingParallel(lengths=lengths, returns=(1.0, 3.0)))

    mean_reward, std_reward = evaluate([environment])

    # Each of the two agents contributes 2 of the 4 episodes, also one-step episodes after a restart.
    assert (mean_reward, std_reward) == (pytest.approx(2.0), pytest.approx(1.0))


def test_unsupported_environment_types_are_rejected():
    with pytest.raises(TypeError, match="not supported"):
        evaluate(["not an environment"])


def test_deterministic_flag_is_passed_to_the_agent():
    agent = ConstantActionAgent()
    evaluate([Toy()], agent=agent, deterministic=True)
    assert agent.deterministic_flags == {True}


def test_features_extractor_preprocesses_observations_before_choosing_actions():
    agent = ConstantActionAgent(features_extractor=Doubling())
    evaluate([Toy(observation_space=ONES)], agent=agent)
    assert all(np.array_equal(observation, [2, 2, 2]) for observation in agent.observations)


def test_evaluation_episode_rewards_are_logged_to_the_connector():
    connector = RecordingConnector()
    evaluate([Toy()], connector=connector, logging_frequency=1)

    logged = connector.value_sequences_to_log["Evaluation - Episode reward"]
    assert logged
    assert all(value == 5.0 for _, value in logged)

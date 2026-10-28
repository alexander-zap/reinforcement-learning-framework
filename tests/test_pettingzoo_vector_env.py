"""PettingZooVectorEnv: one slot per possible agent, auto-reset, dummy steps for agents which leave early."""

import numpy as np
from imitation.algorithms.adversarial.gail import GAIL
from imitation.data.types import TrajectoryWithRew
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from rl_framework.agent.imitation import EpisodeSequence
from rl_framework.util import DummyConnector
from rl_framework.util.pettingzoo_vector_env import (
    PettingZooVectorEnv,
    pettingzoo_environments_to_vec_env,
)
from tests.test_imitation_agent import ALGORITHM_PARAMETERS
from tests.toys import (
    BOX,
    InProcessImitationAgent,
    LeavingParallel,
    RecordingConnector,
    SB3Agent,
    ToyParallel,
)

PPO_PARAMETERS = {"n_steps": 20, "batch_size": 20, "n_epochs": 1, "device": "cpu"}


def step(vector_env):
    return vector_env.step(np.zeros(vector_env.num_envs, dtype=np.int64))


def test_left_agents_get_discarded_truncated_dummy_steps_until_the_reset():
    vector_env = PettingZooVectorEnv(LeavingParallel(episode_length=5, leave_after=2))
    vector_env.reset()

    step(vector_env)
    _, rewards, terminations, truncations, infos = step(vector_env)
    # "b" leaves with its real last step
    assert rewards.tolist() == [2.0, 2.0]
    assert terminations.tolist() == [False, True] and truncations.tolist() == [False, False]
    assert "discard" not in infos[1]

    for _ in range(2):
        observations, rewards, terminations, truncations, infos = step(vector_env)
        assert rewards[1] == 0.0 and not terminations[1] and truncations[1]
        assert infos[1]["discard"] is True
        np.testing.assert_array_equal(observations[1], np.zeros(3, np.float32))
        np.testing.assert_array_equal(infos[1]["terminal_observation"], np.zeros(3, np.float32))
        assert not terminations[0] and not truncations[0] and "discard" not in infos[0]

    # "a" finishes: reset, both slots start the next episode
    observations, rewards, terminations, truncations, infos = step(vector_env)
    assert rewards[0] == 5.0 and terminations[0]
    assert BOX.contains(infos[0]["terminal_observation"])
    assert BOX.contains(observations[0]) and BOX.contains(observations[1])
    assert not np.array_equal(observations[1], np.zeros(3, np.float32))


def test_environments_without_render_mode_are_supported():
    environment = ToyParallel()
    del environment.render_mode

    assert PettingZooVectorEnv(environment).render_mode is None


def test_sb3_training_with_left_agents_sees_only_the_real_episodes():
    agent = SB3Agent(PPO, PPO_PARAMETERS)
    connector = RecordingConnector()

    agent.train(total_timesteps=60, connector=connector, training_environments=[LeavingParallel()])

    returns = [reward for _, reward in connector.value_sequences_to_log["Episode reward"]]
    assert returns
    # Mean over the episodes of each log: only the real returns (15 for "a", 3 for "b"), no dummy episodes
    assert all(3.0 <= reward <= 15.0 for reward in returns)
    assert {episode["l"] for episode in agent.algorithm.ep_info_buffer} == {2, 5}
    assert {episode["r"] for episode in agent.algorithm.ep_info_buffer} == {3.0, 15.0}


def test_evaluation_with_left_agents_counts_their_real_episodes():
    agent = SB3Agent(PPO, PPO_PARAMETERS)
    agent.train(total_timesteps=20, training_environments=[LeavingParallel()])

    mean_reward, _ = agent.evaluate([LeavingParallel()], n_eval_episodes=4, connector=DummyConnector())

    assert mean_reward == (15.0 + 15.0 + 3.0 + 3.0) / 4


def test_vectorized_pettingzoo_episodes_end_at_their_last_step():
    # The imitation agents previously used supersuit's black_death_v3, which reported the episode end one step late
    done_steps = []
    vector_env = pettingzoo_environments_to_vec_env([ToyParallel(episode_length=5)], max_num_cpus=1)
    vector_env.reset()
    for t in range(1, 11):
        _, _, dones, infos = vector_env.step(np.zeros(2, dtype=np.int64))
        if dones.any():
            assert dones.all()
            assert all(info["TimeLimit.truncated"] is False for info in infos)
            done_steps.append(t)
    vector_env.close()

    assert done_steps == [5, 10]


def test_imitation_training_on_pettingzoo_environment_with_left_agents():
    episode_lengths = []

    class EpisodeLengthRecorder(BaseCallback):
        def _on_step(self):
            episode_lengths.extend(info["episode"]["l"] for info in self.locals["infos"] if "episode" in info)
            return True

    class RecordingAgent(InProcessImitationAgent):
        def get_callbacks(self, connector):
            return super().get_callbacks(connector) + [EpisodeLengthRecorder()]

    trajectory = TrajectoryWithRew(
        obs=np.zeros((6, 3), np.float32), acts=np.zeros(5, np.int64), infos=None, terminal=True, rews=np.zeros(5)
    )
    agent = RecordingAgent(GAIL, ALGORITHM_PARAMETERS[GAIL])
    agent.train(64, EpisodeSequence.from_episodes([trajectory] * 13), training_environments=[LeavingParallel()])

    assert episode_lengths and set(episode_lengths) == {2, 5}

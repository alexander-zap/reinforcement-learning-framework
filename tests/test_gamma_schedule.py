import gymnasium
import pytest
import stable_baselines3
from async_gym_agents.agents.async_agent import get_injected_agent
from async_gym_agents.envs.multi_env import IndexableMultiEnv
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.env_util import make_vec_env

from rl_framework.agent.reinforcement.stable_baselines import StableBaselinesAgent
from rl_framework.util import DummyConnector, GammaScheduleCallback

N_STEPS = 64
N_ROLLOUTS = 3


def linear_gamma(progress_remaining: float) -> float:
    return 0.9 + 0.09 * (1.0 - progress_remaining)


class RecordRolloutGamma(BaseCallback):
    """Records the gamma that each rollout's returns and advantages were computed with."""

    def __init__(self):
        super().__init__()
        self.model_gammas = []
        self.buffer_gammas = []

    def _on_rollout_end(self) -> None:
        self.model_gammas.append(self.model.gamma)
        self.buffer_gammas.append(self.model.rollout_buffer.gamma)

    def _on_step(self) -> bool:
        return True

    def process_episode(self, context) -> bool:
        return True

    def advance_callback(self, transition_count: int, num_timesteps: int) -> None:
        self.n_calls += transition_count
        self.num_timesteps = num_timesteps


def expected_gammas(total_timesteps: int) -> list[float]:
    return [linear_gamma(1.0 - rollout * N_STEPS / total_timesteps) for rollout in range(N_ROLLOUTS)]


def test_agent_replaces_callable_gamma_with_start_value():
    agent = StableBaselinesAgent(algorithm_parameters={"gamma": linear_gamma})

    assert agent.gamma_schedule is linear_gamma
    assert agent.algorithm_parameters["gamma"] == pytest.approx(0.9)
    assert any(
        isinstance(callback, GammaScheduleCallback) for callback in agent.get_callbacks(connector=DummyConnector())
    )


def test_agent_without_gamma_schedule_adds_no_callback():
    agent = StableBaselinesAgent(algorithm_parameters={"gamma": 0.95})

    assert agent.gamma_schedule is None
    assert agent.algorithm_parameters["gamma"] == 0.95
    assert not any(
        isinstance(callback, GammaScheduleCallback) for callback in agent.get_callbacks(connector=DummyConnector())
    )


def test_callback_sets_gamma_before_each_rollout():
    env = make_vec_env("CartPole-v1", n_envs=1)
    model = stable_baselines3.PPO("MlpPolicy", env, n_steps=N_STEPS, batch_size=N_STEPS, n_epochs=1, gamma=0.5)
    recorder = RecordRolloutGamma()
    total_timesteps = N_STEPS * N_ROLLOUTS

    model.learn(total_timesteps, callback=CallbackList([GammaScheduleCallback(linear_gamma), recorder]))

    assert recorder.model_gammas == pytest.approx(expected_gammas(total_timesteps))
    assert recorder.buffer_gammas == pytest.approx(expected_gammas(total_timesteps))


def test_callback_sets_gamma_for_async_on_policy_rollouts():
    env = IndexableMultiEnv([lambda: gymnasium.make("CartPole-v1") for _ in range(2)])
    model = get_injected_agent(stable_baselines3.PPO)(
        "MlpPolicy", env, n_steps=N_STEPS, batch_size=N_STEPS, n_epochs=1, gamma=0.5, device="cpu"
    )
    recorder = RecordRolloutGamma()
    try:
        model.learn(N_STEPS * N_ROLLOUTS, callback=CallbackList([GammaScheduleCallback(linear_gamma), recorder]))
    finally:
        model.shutdown()

    # Async rollouts can overshoot n_steps (whole episodes are assembled), so check the schedule is followed rather
    # than exact values.
    assert recorder.model_gammas == recorder.buffer_gammas
    assert recorder.model_gammas[0] == pytest.approx(0.9)
    assert all(later > earlier for earlier, later in zip(recorder.model_gammas, recorder.model_gammas[1:]))

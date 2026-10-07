"""reset_optimizer_state: which optimizers are cleared for each supported SB3 algorithm."""

from types import SimpleNamespace

import pytest
import sb3_contrib
import stable_baselines3
import torch as th
from gymnasium import spaces

from rl_framework.util import reset_optimizer_state
from rl_framework.util.sb3_optimizer_reset import get_optimizers_to_reset
from tests.toys import Toy

DISCRETE = Toy()
CONTINUOUS = Toy(action_space=spaces.Box(-1, 1, (2,)))


@pytest.mark.parametrize(
    "algorithm_class, environment, expected_paths",
    [
        (stable_baselines3.A2C, DISCRETE, ["policy.optimizer"]),
        (stable_baselines3.PPO, DISCRETE, ["policy.optimizer"]),
        (stable_baselines3.DQN, DISCRETE, ["policy.optimizer"]),
        (sb3_contrib.TRPO, DISCRETE, ["policy.optimizer"]),
        (stable_baselines3.DDPG, CONTINUOUS, ["policy.actor.optimizer", "policy.critic.optimizer"]),
        (stable_baselines3.TD3, CONTINUOUS, ["policy.actor.optimizer", "policy.critic.optimizer"]),
        (
            stable_baselines3.SAC,
            CONTINUOUS,
            ["ent_coef_optimizer", "policy.actor.optimizer", "policy.critic.optimizer"],
        ),
    ],
)
def test_reset_clears_the_state_of_every_optimizer(algorithm_class, environment, expected_paths):
    algorithm = algorithm_class("MlpPolicy", environment)
    optimizers = get_optimizers_to_reset(algorithm)
    for optimizer in optimizers.values():
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                optimizer.state[parameter] = {"step": th.tensor(1.0)}

    assert reset_optimizer_state(algorithm) == expected_paths
    assert all(not optimizer.state for optimizer in optimizers.values())


def test_sac_with_fixed_entropy_coefficient_has_no_entropy_optimizer():
    algorithm = stable_baselines3.SAC("MlpPolicy", CONTINUOUS, ent_coef=0.1)
    assert sorted(get_optimizers_to_reset(algorithm)) == ["policy.actor.optimizer", "policy.critic.optimizer"]


def test_unsupported_algorithms_are_rejected():
    algorithm = sb3_contrib.QRDQN("MlpPolicy", DISCRETE)
    with pytest.raises(NotImplementedError, match="QRDQN"):
        reset_optimizer_state(algorithm)


def test_missing_or_invalid_optimizers_are_reported():
    ppo = stable_baselines3.PPO("MlpPolicy", DISCRETE)
    ppo.policy.optimizer = SimpleNamespace()
    with pytest.raises(RuntimeError, match="invalid optimizer at 'policy.optimizer'"):
        reset_optimizer_state(ppo)

    del ppo.policy.optimizer
    with pytest.raises(RuntimeError, match="missing required optimizer attribute 'policy.optimizer'"):
        reset_optimizer_state(ppo)

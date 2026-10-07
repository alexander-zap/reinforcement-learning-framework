"""The initial_action_bias algorithm parameter: validation and the real SB3/AsyncSB3 training/save-load lifecycle."""

import io

import gymnasium as gym
import numpy as np
import pytest
import sb3_contrib
import stable_baselines3
import torch as th
from async_gym_agents.callback_batching import CallbackBatchDispatcher
from async_gym_agents.policy_transport import SharedPolicyStore
from gymnasium import spaces
from stable_baselines3.common.callbacks import CallbackList

from rl_framework.agent.reinforcement.async_stable_baselines import (
    AsyncStableBaselinesAgent,
)
from rl_framework.agent.reinforcement.stable_baselines import StableBaselinesAgent
from rl_framework.util import DummyConnector, validate_initial_action_bias
from rl_framework.util.policy_init import _action_output_layers, apply_action_bias

SPACE = spaces.MultiDiscrete([3, 3])
INIT_BIAS = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
BOX = spaces.Box(-1, 1, (2,), np.float32)


class Toy(gym.Env):
    def __init__(self, action_space=SPACE):
        self.observation_space = spaces.Box(-1, 1, (4,), np.float32)
        self.action_space = action_space

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(4, np.float32), {}

    def step(self, action):
        return np.zeros(4, np.float32), 0.0, True, False, {}


def agent_parameters(agent_class=AsyncStableBaselinesAgent, **params):
    async_params = {"use_mp": False, "worker_join_timeout": 5.0} if agent_class is AsyncStableBaselinesAgent else {}
    return {
        "policy": "MlpPolicy",
        "device": "cpu",
        "seed": 0,
        "n_steps": 8,
        "batch_size": 4,
        "n_epochs": 1,
        # Keep parameters unchanged by the optimiser so lifecycle mutations are observable.
        "learning_rate": 0.0,
        "tensorboard_log": None,
        "policy_kwargs": {"net_arch": {"pi": [16], "vf": [16]}},
        **async_params,
        **params,
    }


def make_agent(agent_class=AsyncStableBaselinesAgent, **params):
    return agent_class(
        algorithm_class=stable_baselines3.PPO, algorithm_parameters=agent_parameters(agent_class, **params)
    )


def train_sb3(agent, action_space=SPACE):
    """Train the (synchronous) SB3 agent; with learning rate 0, its policy afterwards is the one it started with."""
    agent.train(total_timesteps=8, connector=DummyConnector(), training_environments=[Toy(action_space)])
    return {name: tensor.cpu() for name, tensor in agent.algorithm.policy.state_dict().items()}


@pytest.fixture
def train_snapshot(monkeypatch):
    """Run the real async workers and inspect the first state dict published to them."""
    snapshots = []
    original_create = SharedPolicyStore.create

    def capture(**kwargs):
        snapshots.append(th.load(io.BytesIO(kwargs["initial_payload"]), map_location="cpu", weights_only=True))
        return original_create(**kwargs)

    monkeypatch.setattr(SharedPolicyStore, "create", capture)

    def train(agent, action_space=SPACE):
        count_before = len(snapshots)
        try:
            agent.train(total_timesteps=8, connector=DummyConnector(), training_environments=[Toy(action_space)])
        finally:
            agent.algorithm.shutdown()
            if agent.algorithm.get_env() is not None:
                agent.algorithm.get_env().close()
        assert len(snapshots) == count_before + 1, "expected one initial publication per training call"
        return snapshots[-1]

    return train


@pytest.mark.parametrize(
    "bad, message",
    [
        ([0, 0, float("inf")], "finite numbers"),
        ([[0, 0, float("nan")], [0]], "finite numbers"),
        ([0, 0, "x"], "sequence of numbers"),
        (3, "sequence of numbers"),
        ([], "nonempty sequence"),
        ([[0, 0, 1], []], "nonempty sequence"),
        ([[0, 0, 1], 0], "sequence of numbers"),
    ],
)
def test_invalid_biases_are_rejected(bad, message):
    with pytest.raises(ValueError, match=message):
        validate_initial_action_bias(bad, stable_baselines3.PPO)


def test_bias_is_flat_or_one_sequence_per_action_head():
    assert validate_initial_action_bias([0, -1.5, 1], stable_baselines3.PPO) == pytest.approx([0, -1.5, 1])
    # One sequence per action head, joined in order.
    assert validate_initial_action_bias([[0, 1], (2, 3, 4)], stable_baselines3.PPO) == pytest.approx([0, 1, 2, 3, 4])
    assert validate_initial_action_bias(None, stable_baselines3.PPO) is None


@pytest.mark.parametrize(
    "device, ortho_init",
    [
        ("cpu", True),
        ("cpu", False),
        pytest.param("cuda", True, marks=pytest.mark.skipif(not th.cuda.is_available(), reason="needs CUDA")),
    ],
)
def test_first_async_snapshot_changes_only_the_action_bias(train_snapshot, device, ortho_init):
    params = {"device": device, "policy_kwargs": {"net_arch": {"pi": [16], "vf": [16]}, "ortho_init": ortho_init}}
    baseline = train_snapshot(make_agent(**params))
    agent = make_agent(initial_action_bias=INIT_BIAS, **params)
    initial = train_snapshot(agent)

    baseline["action_net.bias"] = th.tensor(INIT_BIAS)
    th.testing.assert_close(initial, baseline)
    if ortho_init:
        # These probabilities are exact for zero network output at initialisation only.
        with th.no_grad():
            movement, rotation = agent.algorithm.policy.get_distribution(th.zeros(1, 4, device=device)).distribution
        assert movement.probs.cpu().numpy()[0] == pytest.approx([0.2119, 0.2119, 0.5761], abs=1e-3)
        assert rotation.probs.cpu().numpy()[0] == pytest.approx([1 / 3] * 3, abs=1e-3)


@pytest.mark.parametrize(
    "action_space, action_bias, expected",
    [
        (spaces.MultiDiscrete([5, 3]), [[0, 0, 0, 0, 1], [0.5, 0, 0]], [0, 0, 0, 0, 1, 0.5, 0, 0]),
        (spaces.Discrete(4), [0, 0, 1, -1], [0, 0, 1, -1]),
        (spaces.Box(-1, 1, (2,), np.float32), [0.25, -0.5], [0.25, -0.5]),  # Offsets of the action mean.
    ],
    ids=["multi-discrete", "discrete", "box"],
)
def test_bias_is_set_for_any_action_space(train_snapshot, action_space, action_bias, expected):
    initial = train_snapshot(make_agent(initial_action_bias=action_bias), action_space)

    assert initial["action_net.bias"].numpy() == pytest.approx(expected)


@pytest.mark.parametrize("action_space", [spaces.MultiDiscrete([5, 3]), spaces.Box(-1, 1, (2,), np.float32)])
def test_bias_of_the_wrong_length_is_rejected(train_snapshot, action_space):
    with pytest.raises(ValueError, match="initial_action_bias has 6 entries but the policy's action output layer"):
        train_snapshot(make_agent(initial_action_bias=INIT_BIAS), action_space)


@pytest.mark.parametrize("zero_bias", [True, False])
def test_loaded_checkpoints_are_unchanged_even_with_zero_bias(train_snapshot, tmp_path, zero_bias):
    source = make_agent()
    train_snapshot(source)
    if not zero_bias:
        with th.no_grad():
            source.algorithm.policy.action_net.bias.copy_(th.tensor([0.3, -0.2, 0.1, -0.1, 0.2, -0.3]))
    expected = source.algorithm.policy.state_dict()
    assert source.algorithm.num_timesteps > 0
    if zero_bias:
        assert th.count_nonzero(expected["action_net.bias"]) == 0
    checkpoint = tmp_path / "checkpoint.zip"
    source.save_to_file(checkpoint)

    resumed = make_agent(initial_action_bias=INIT_BIAS)
    resumed.load_from_file(checkpoint)
    th.testing.assert_close(train_snapshot(resumed), expected)


def test_repeated_training_preserves_updated_bias(train_snapshot):
    agent = make_agent(initial_action_bias=INIT_BIAS)
    train_snapshot(agent)
    with th.no_grad():
        agent.algorithm.policy.action_net.bias.copy_(th.tensor([0.3, -0.2, 0.1, -0.1, 0.2, -0.3]))
    expected = agent.algorithm.policy.state_dict()
    th.testing.assert_close(train_snapshot(agent), expected)


def test_failed_training_leaves_the_bias_for_the_next_training(train_snapshot):
    agent = make_agent(initial_action_bias=INIT_BIAS)
    with pytest.raises(ValueError, match="No training environments"):
        agent.train(training_environments=[])
    initial = train_snapshot(agent)
    assert initial["action_net.bias"].numpy() == pytest.approx(INIT_BIAS)


@pytest.mark.parametrize("initial_action_bias", [None, INIT_BIAS])
def test_parameter_is_consumed_without_adding_per_step_callbacks(initial_action_bias):
    agent = make_agent(initial_action_bias=initial_action_bias)
    assert "initial_action_bias" not in agent.algorithm_parameters, "it must not reach the SB3 constructor"
    callbacks = CallbackList(agent.get_callbacks(DummyConnector()))
    assert CallbackBatchDispatcher(callbacks).step_callbacks == []


# --- Synchronous SB3 agent -------------------------------------------------------------------------


def test_sb3_agent_initialises_only_the_action_bias_of_a_fresh_model():
    baseline = train_sb3(make_agent(StableBaselinesAgent))
    initial = train_sb3(make_agent(StableBaselinesAgent, initial_action_bias=INIT_BIAS))

    baseline["action_net.bias"] = th.tensor(INIT_BIAS)
    th.testing.assert_close(initial, baseline)


def test_sb3_agent_leaves_loaded_checkpoints_and_retrained_models_unchanged(tmp_path):
    source = make_agent(StableBaselinesAgent)
    train_sb3(source)
    with th.no_grad():
        source.algorithm.policy.action_net.bias.copy_(th.tensor([0.3, -0.2, 0.1, -0.1, 0.2, -0.3]))
    expected = {name: tensor.cpu() for name, tensor in source.algorithm.policy.state_dict().items()}
    checkpoint = tmp_path / "checkpoint.zip"
    source.save_to_file(checkpoint)

    resumed = make_agent(StableBaselinesAgent, initial_action_bias=INIT_BIAS)
    # As a download does, with the algorithm parameters (including initial_action_bias) passed again.
    resumed.load_from_file(checkpoint, agent_parameters(StableBaselinesAgent, initial_action_bias=INIT_BIAS))
    assert "initial_action_bias" not in resumed.algorithm_parameters
    th.testing.assert_close(train_sb3(resumed), expected)
    # Training the same agent again must not re-apply the bias either.
    th.testing.assert_close(train_sb3(resumed), expected)


def test_sb3_agent_rejects_a_bias_of_the_wrong_length():
    with pytest.raises(ValueError, match="initial_action_bias has 6 entries"):
        train_sb3(make_agent(StableBaselinesAgent, initial_action_bias=INIT_BIAS), spaces.Box(-1, 1, (2,), np.float32))


@pytest.mark.parametrize("algorithm_class", [stable_baselines3.DQN, sb3_contrib.QRDQN])
@pytest.mark.parametrize("agent_class", [StableBaselinesAgent, AsyncStableBaselinesAgent])
def test_q_value_algorithms_are_rejected_with_the_reason(agent_class, algorithm_class):
    with pytest.raises(ValueError, match=f"does not support {algorithm_class.__name__}: it acts on the argmax"):
        agent_class(
            algorithm_class=algorithm_class,
            algorithm_parameters={"policy": "MlpPolicy", "device": "cpu", "initial_action_bias": INIT_BIAS},
        )


def test_ars_is_rejected_for_discrete_action_spaces():
    agent = StableBaselinesAgent(
        algorithm_class=sb3_contrib.ARS,
        algorithm_parameters={"policy": "MlpPolicy", "device": "cpu", "initial_action_bias": [0, 0, 1]},
    )
    with pytest.raises(ValueError, match="does not support ARS with a Discrete action space"):
        agent.train(total_timesteps=8, connector=DummyConnector(), training_environments=[Toy(spaces.Discrete(3))])


def test_ars_gets_the_bias_for_box_action_spaces():
    # Applied directly: ARS does not train with rl_framework's callbacks yet.
    algorithm = sb3_contrib.ARS("MlpPolicy", Toy(BOX), device="cpu")
    apply_action_bias(algorithm, np.array([0.5, -0.25]))

    assert _action_output_layers(algorithm.policy)[0].bias.detach().numpy() == pytest.approx([0.5, -0.25])


# --- SAC and TD3: the actor's mean before tanh --------------------------------------------------------


def make_off_policy_agent(algorithm_class, **params):
    return StableBaselinesAgent(
        algorithm_class=algorithm_class,
        algorithm_parameters={
            "policy": "MlpPolicy",
            "device": "cpu",
            "seed": 0,
            "buffer_size": 100,
            # No gradient step within the few training steps of a test, so the policy stays as it was initialised.
            "learning_starts": 1000,
            "tensorboard_log": None,
            "policy_kwargs": {"net_arch": [16]},
            **params,
        },
    )


def actor_mean_bias(actor):
    return [module for module in actor.mu.modules() if isinstance(module, th.nn.Linear)][-1].bias.detach().numpy()


@pytest.mark.parametrize("algorithm_class", [stable_baselines3.SAC, stable_baselines3.TD3])
def test_actor_mean_of_a_fresh_off_policy_model_gets_the_bias(algorithm_class):
    agent = make_off_policy_agent(algorithm_class, initial_action_bias=[0.5, -0.25])
    agent.train(total_timesteps=8, connector=DummyConnector(), training_environments=[Toy(BOX)])

    actors = [agent.algorithm.policy.actor, getattr(agent.algorithm.policy, "actor_target", None)]
    for actor in filter(None, actors):
        assert actor_mean_bias(actor) == pytest.approx([0.5, -0.25])
    # Before tanh: the deterministic action for a near-zero network output is about tanh(bias).
    action, _ = agent.algorithm.predict(np.zeros(4, np.float32), deterministic=True)
    assert action == pytest.approx(np.tanh([0.5, -0.25]), abs=0.1)


def test_off_policy_bias_of_the_wrong_length_is_rejected():
    agent = make_off_policy_agent(stable_baselines3.SAC, initial_action_bias=[0.5, -0.25, 0.0])
    with pytest.raises(ValueError, match="initial_action_bias has 3 entries but the policy's action output layer"):
        agent.train(total_timesteps=8, connector=DummyConnector(), training_environments=[Toy(BOX)])


class MaskableToy(Toy):
    def action_masks(self):
        return np.ones(int(self.action_space.nvec.sum()), dtype=bool)


ON_POLICY = {"n_steps": 8, "batch_size": 8, "learning_rate": 0.0}
OFF_POLICY = {"buffer_size": 100, "learning_starts": 1000}


@pytest.mark.parametrize(
    "algorithm_class, policy, env, action_bias, params",
    [
        (stable_baselines3.A2C, "MlpPolicy", Toy(), INIT_BIAS, {"n_steps": 8, "learning_rate": 0.0}),
        (sb3_contrib.TRPO, "MlpPolicy", Toy(), INIT_BIAS, ON_POLICY),
        (sb3_contrib.MaskablePPO, "MlpPolicy", MaskableToy(), INIT_BIAS, ON_POLICY),
        (sb3_contrib.RecurrentPPO, "MlpLstmPolicy", Toy(), INIT_BIAS, ON_POLICY),
        (stable_baselines3.DDPG, "MlpPolicy", Toy(BOX), [0.5, -0.25], OFF_POLICY),
        (sb3_contrib.TQC, "MlpPolicy", Toy(BOX), [0.5, -0.25], OFF_POLICY),
        (sb3_contrib.CrossQ, "MlpPolicy", Toy(BOX), [0.5, -0.25], OFF_POLICY),
    ],
    ids=lambda value: value.__name__ if isinstance(value, type) else None,
)
def test_bias_reaches_the_action_output_layer_of_each_supported_algorithm(
    algorithm_class, policy, env, action_bias, params
):
    agent = StableBaselinesAgent(
        algorithm_class=algorithm_class,
        algorithm_parameters={
            "policy": policy,
            "device": "cpu",
            "seed": 0,
            "tensorboard_log": None,
            "initial_action_bias": action_bias,
            **params,
        },
    )
    agent.train(total_timesteps=8, connector=DummyConnector(), training_environments=[env])

    layers = _action_output_layers(agent.algorithm.policy)
    assert layers, "expected an action output layer"
    for layer in layers:
        assert layer.bias.detach().numpy() == pytest.approx(action_bias)

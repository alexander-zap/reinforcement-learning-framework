"""ONNX export of SB3 policies: exported contract and parity with the policies (onnx' reference evaluator)."""

import numpy as np
import onnx
import pytest
import torch as th
from gymnasium import spaces
from onnx.reference import ReferenceEvaluator
from stable_baselines3 import DQN, PPO, SAC, TD3

from rl_framework.util.onnx_export import GUMBEL_EPSILON, export_sb3_policy_as_onnx
from tests.toys import BOX, Toy

BATCH = 5
RANDOM_OP_TYPES = {"Multinomial", "RandomNormal", "RandomNormalLike", "RandomUniform", "RandomUniformLike"}


def build(algorithm_class, observation_space=BOX, action_space=spaces.Discrete(3), **parameters):
    policy = "MultiInputPolicy" if isinstance(observation_space, spaces.Dict) else "MlpPolicy"
    return algorithm_class(policy, Toy(observation_space, action_space), device="cpu", seed=0, **parameters)


def export(model, tmp_path, deterministic=True):
    file_path = tmp_path / "policy.onnx"
    export_sb3_policy_as_onnx(model.policy, file_path, deterministic=deterministic)
    onnx_model = onnx.load(file_path)
    onnx.checker.check_model(onnx_model)
    return file_path, onnx_model


def sample_observations(observation_space, rng):
    observation_space.seed(int(rng.integers(1 << 31)))
    samples = [observation_space.sample() for _ in range(BATCH)]
    if isinstance(observation_space, spaces.Dict):
        return {key: np.stack([sample[key] for sample in samples]) for key in observation_space.spaces}
    return np.stack(samples)


def onnx_inputs(model, observations, noise=None, noise_name=None):
    """
    Feeds of the exported model: float32 observations (per key for Dict observations), optional noise.
    The observations are laid out like the policy's observation space (e.g., images channel-first, as SB3's `predict`
    transposes them), via `obs_to_tensor`.
    """
    tensor = observation_tensor(model, observations)
    inputs = (
        {key: value.numpy().astype(np.float32) for key, value in tensor.items()}
        if isinstance(tensor, dict)
        else {"obs": tensor.numpy().astype(np.float32)}
    )
    if noise is not None:
        inputs[noise_name] = noise.astype(np.float32)
    return inputs


def run(file_path, inputs):
    session = ReferenceEvaluator(str(file_path))
    return dict(zip(session.output_names, session.run(None, inputs)))


def observation_tensor(model, observations):
    tensor, _ = model.policy.obs_to_tensor(observations)
    return tensor


# --- contract ---


@pytest.mark.parametrize(
    "algorithm_class, action_space, deterministic, input_names, output_names",
    [
        (PPO, spaces.Discrete(3), True, ["obs"], ["action", "values", "log_prob"]),
        (PPO, spaces.Discrete(3), False, ["obs", "noise_u"], ["action", "values", "log_prob"]),
        (PPO, spaces.Box(-1, 1, (2,)), False, ["obs", "noise_eps"], ["action", "values", "log_prob"]),
        (SAC, spaces.Box(-1, 1, (2,)), True, ["obs"], ["action", "log_prob"]),
        (SAC, spaces.Box(-1, 1, (2,)), False, ["obs", "noise_eps"], ["action", "log_prob"]),
        (TD3, spaces.Box(-1, 1, (2,)), True, ["obs"], ["action"]),
        (DQN, spaces.Discrete(3), True, ["obs"], ["action", "q_values"]),
    ],
)
def test_exported_inputs_and_outputs_are_named_float_and_batched_without_random_nodes(
    tmp_path, algorithm_class, action_space, deterministic, input_names, output_names
):
    _, model = export(build(algorithm_class, action_space=action_space), tmp_path, deterministic)

    assert [graph_input.name for graph_input in model.graph.input] == input_names
    assert [graph_output.name for graph_output in model.graph.output] == output_names
    assert all(graph_input.type.tensor_type.elem_type == onnx.TensorProto.FLOAT for graph_input in model.graph.input)
    for value in list(model.graph.input) + list(model.graph.output):
        assert value.type.tensor_type.shape.dim[0].dim_param == "batch", value.name
    assert not {node.op_type for node in model.graph.node} & RANDOM_OP_TYPES


def test_dict_observations_are_exported_as_one_input_per_key(tmp_path):
    observation_space = spaces.Dict({"position": BOX, "velocity": spaces.Box(-1, 1, (2,))})
    _, model = export(build(PPO, observation_space), tmp_path)

    assert [graph_input.name for graph_input in model.graph.input] == ["position", "velocity"]


# --- parity: actor-critic ---

OBSERVATION_SPACES = {
    "box": BOX,
    "discrete": spaces.Discrete(5),
    "multi-discrete": spaces.MultiDiscrete([3, 4]),
    "uint8-image": spaces.Box(0, 255, (36, 36, 3), np.uint8),
    "dict": spaces.Dict({"position": BOX, "velocity": spaces.Box(-1, 1, (2,))}),
}


@pytest.mark.parametrize("observation_space", OBSERVATION_SPACES.values(), ids=OBSERVATION_SPACES.keys())
@pytest.mark.parametrize(
    "action_space",
    [spaces.Discrete(3), spaces.MultiDiscrete([3, 2]), spaces.Box(-0.5, 0.5, (2,))],
    ids=["discrete", "multi-discrete", "box"],
)
def test_deterministic_actor_critic_export_matches_the_policy(tmp_path, observation_space, action_space):
    model = build(PPO, observation_space, action_space)
    file_path, _ = export(model, tmp_path)
    observations = sample_observations(observation_space, np.random.default_rng(0))

    outputs = run(file_path, onnx_inputs(model, observations))

    expected_actions, _ = model.predict(observations, deterministic=True)
    obs = observation_tensor(model, observations)
    with th.no_grad():
        distribution = model.policy.get_distribution(obs)
        raw_actions = distribution.get_actions(deterministic=True)
        expected_log_prob = distribution.log_prob(raw_actions).numpy()
        expected_values = model.policy.predict_values(obs).numpy()
    np.testing.assert_allclose(
        outputs["action"].reshape(expected_actions.shape), expected_actions, rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(outputs["values"], expected_values, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(outputs["log_prob"], expected_log_prob, rtol=1e-4, atol=1e-5)


def test_discrete_actions_have_one_column_per_action_head(tmp_path):
    rng = np.random.default_rng(0)
    for action_space, heads in [(spaces.Discrete(3), 1), (spaces.MultiDiscrete([3, 2, 4]), 3)]:
        model = build(PPO, action_space=action_space)
        file_path, _ = export(model, tmp_path)
        action = run(file_path, onnx_inputs(model, sample_observations(BOX, rng)))["action"]
        assert action.shape == (BATCH, heads)
        assert action.dtype == np.int64


@pytest.mark.parametrize(
    "action_space", [spaces.Discrete(3), spaces.MultiDiscrete([3, 2])], ids=["discrete", "multi-discrete"]
)
def test_stochastic_discrete_export_samples_with_gumbel_max_from_the_noise(tmp_path, action_space):
    model = build(PPO, action_space=action_space)
    file_path, _ = export(model, tmp_path, deterministic=False)
    rng = np.random.default_rng(0)
    observations = sample_observations(BOX, rng)
    head_dims = [action_space.n] if isinstance(action_space, spaces.Discrete) else list(action_space.nvec)
    noise_u = rng.uniform(size=(BATCH, sum(head_dims)))

    outputs = run(file_path, onnx_inputs(model, observations, noise_u, "noise_u"))

    with th.no_grad():
        distribution = model.policy.get_distribution(observation_tensor(model, observations))
        head_distributions = (
            [distribution.distribution] if isinstance(action_space, spaces.Discrete) else distribution.distribution
        )
        gumbel = -np.log(-np.log(np.clip(noise_u, GUMBEL_EPSILON, 1 - GUMBEL_EPSILON)))
        gumbel_heads = np.split(gumbel, np.cumsum(head_dims)[:-1], axis=1)
        expected_actions = np.stack(
            [
                np.argmax(head.logits.numpy() + head_gumbel, axis=1)
                for head, head_gumbel in zip(head_distributions, gumbel_heads)
            ],
            axis=1,
        )
        actions_tensor = th.as_tensor(expected_actions.squeeze(1) if len(head_dims) == 1 else expected_actions)
        expected_log_prob = distribution.log_prob(actions_tensor).numpy()
    np.testing.assert_array_equal(outputs["action"], expected_actions)
    np.testing.assert_allclose(outputs["log_prob"], expected_log_prob, rtol=1e-4, atol=1e-5)


def test_stochastic_box_export_samples_with_reparameterization_and_clips_the_action(tmp_path):
    model = build(PPO, action_space=spaces.Box(-0.5, 0.5, (2,)))
    file_path, _ = export(model, tmp_path, deterministic=False)
    rng = np.random.default_rng(0)
    observations = sample_observations(BOX, rng)
    noise_eps = rng.normal(size=(BATCH, 2)) * 3  # large enough to exceed the action bounds

    outputs = run(file_path, onnx_inputs(model, observations, noise_eps, "noise_eps"))

    with th.no_grad():
        distribution = model.policy.get_distribution(observation_tensor(model, observations))
        gaussian = distribution.distribution
        raw_actions = gaussian.mean + gaussian.stddev * th.as_tensor(noise_eps, dtype=th.float32)
        expected_log_prob = distribution.log_prob(raw_actions).numpy()
    np.testing.assert_allclose(outputs["action"], np.clip(raw_actions.numpy(), -0.5, 0.5), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(outputs["log_prob"], expected_log_prob, rtol=1e-4, atol=1e-4)


# --- parity: SAC, TD3, DQN ---


@pytest.mark.parametrize("deterministic", [True, False], ids=["deterministic", "stochastic"])
def test_sac_export_matches_the_policy(tmp_path, deterministic):
    model = build(SAC, action_space=spaces.Box(np.array([-2.0, 0.0]), np.array([2.0, 1.0])))
    file_path, _ = export(model, tmp_path, deterministic)
    rng = np.random.default_rng(0)
    observations = sample_observations(BOX, rng)
    noise_eps = None if deterministic else rng.normal(size=(BATCH, 2))

    outputs = run(file_path, onnx_inputs(model, observations, noise_eps, "noise_eps"))

    actor = model.policy.actor
    with th.no_grad():
        mean, log_std, _ = actor.get_action_dist_params(observation_tensor(model, observations))
        gaussian_actions = mean if deterministic else mean + log_std.exp() * th.as_tensor(noise_eps, dtype=th.float32)
        squashed_actions = th.tanh(gaussian_actions)
        distribution = actor.action_dist.proba_distribution(mean, log_std)
        expected_log_prob = distribution.log_prob(squashed_actions, gaussian_actions).numpy()
    expected_actions = model.policy.unscale_action(squashed_actions.numpy())
    if deterministic:
        np.testing.assert_allclose(expected_actions, model.predict(observations, deterministic=True)[0], atol=1e-6)
    np.testing.assert_allclose(outputs["action"], expected_actions, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(outputs["log_prob"], expected_log_prob, rtol=1e-4, atol=1e-4)


def test_td3_export_matches_the_policy(tmp_path):
    model = build(TD3, action_space=spaces.Box(np.array([-2.0, 0.0]), np.array([2.0, 1.0])))
    file_path, _ = export(model, tmp_path)
    observations = sample_observations(BOX, np.random.default_rng(0))

    outputs = run(file_path, onnx_inputs(model, observations))

    expected_actions, _ = model.predict(observations, deterministic=True)
    np.testing.assert_allclose(outputs["action"], expected_actions, rtol=1e-5, atol=1e-5)


def test_dqn_export_matches_the_policy(tmp_path):
    model = build(DQN)
    file_path, _ = export(model, tmp_path)
    observations = sample_observations(BOX, np.random.default_rng(0))

    outputs = run(file_path, onnx_inputs(model, observations))

    with th.no_grad():
        expected_q_values = model.policy.q_net(observation_tensor(model, observations)).numpy()
    np.testing.assert_array_equal(outputs["action"][:, 0], model.predict(observations, deterministic=True)[0])
    np.testing.assert_allclose(outputs["q_values"], expected_q_values, rtol=1e-5, atol=1e-5)


# --- unsupported ---


@pytest.mark.parametrize("algorithm_class", [TD3, DQN])
def test_stochastic_export_of_policies_without_policy_side_sampling_is_rejected(tmp_path, algorithm_class):
    action_space = spaces.Discrete(3) if algorithm_class is DQN else spaces.Box(-1, 1, (2,))
    with pytest.raises(ValueError, match="deterministic=True"):
        export(build(algorithm_class, action_space=action_space), tmp_path, deterministic=False)


@pytest.mark.parametrize(
    "parameters, action_space",
    [
        ({"use_sde": True}, spaces.Box(-1, 1, (2,))),
        ({}, spaces.MultiBinary(2)),
    ],
    ids=["gsde", "multi-binary"],
)
def test_unsupported_actor_critic_policies_are_rejected(tmp_path, parameters, action_space):
    with pytest.raises(NotImplementedError):
        export(build(PPO, action_space=action_space, **parameters), tmp_path)


def test_export_requires_onnx_file_ending(tmp_path):
    with pytest.raises(AssertionError, match=".onnx"):
        export_sb3_policy_as_onnx(build(PPO).policy, tmp_path / "policy.zip")

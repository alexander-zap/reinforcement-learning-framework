"""
ONNX export of Stable-Baselines3 policies (used by the SB3 and imitation agents).

The exported contract follows hfm-inference-server's flat ONNX wrappers (`onnx_export/sb3.py`), extended by the
values and log-probabilities of actor-critic policies:

- Inputs (all float32, dynamic batch axis `batch`):
    - `obs`: the observation; for Dict observation spaces one input per key, named like the key.
      Discrete/MultiDiscrete/image observations are fed as float too (the graph preprocesses them like SB3).
      Observations are laid out like the policy's observation space, e.g., images channel-first (SB3 transposes
      channel-last images in `predict` and with `VecTransposeImage` in training).
    - Stochastic export only: external noise instead of random number nodes, so that sampling is reproducible:
      `noise_u` (Uniform(0, 1), one value per action logit; Gumbel-max sampling) for discrete action spaces,
      `noise_eps` (Normal(0, 1), one value per action dimension; reparameterization) for Box action spaces.
- Outputs (dynamic batch axis `batch`):
    - `action`: discrete actions as int64 of shape (batch, number of action heads), i.e. (batch, 1) for Discrete;
      Box actions postprocessed like SB3's `predict` (clipped to the action space, or unscaled for squashing policies).
    - Actor-critic policies (PPO, A2C, TRPO): `values` (batch, 1) and `log_prob` (batch,) of the (unpostprocessed)
      action, like SB3's `policy.forward`.
    - SAC: `log_prob` (batch,) of the squashed action. DQN: `q_values` (batch, number of actions).
- The graph includes the observation preprocessing and the features extractor, but not the agent's
  `FeaturesExtractor.preprocess` (applied to the environment's observations before the policy).
"""

import math
from pathlib import Path
from typing import List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.distributions import (
    CategoricalDistribution,
    DiagGaussianDistribution,
    MultiCategoricalDistribution,
)
from stable_baselines3.common.policies import ActorCriticPolicy, BasePolicy
from stable_baselines3.dqn.policies import DQNPolicy
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.td3.policies import TD3Policy

# Clamping of the uniform noise for the Gumbel-max trick (like hfm-inference-server)
GUMBEL_EPSILON = 1e-6
# Epsilon of SB3's squashed Gaussian log-probability (`SquashedDiagGaussianDistribution.log_prob`)
SQUASH_EPSILON = 1e-6


def export_sb3_policy_as_onnx(policy: BasePolicy, file_path: Path, deterministic: bool = True) -> None:
    """
    Export an SB3 policy to ONNX (see the module docstring for the exported inputs and outputs).

    Supported policies: actor-critic policies (PPO, A2C, TRPO) with Discrete, MultiDiscrete or Box action spaces
    (without gSDE), SAC, TD3/DDPG (deterministic only) and DQN (deterministic only).

    Args:
        policy: SB3 policy to export.
        file_path: The `.onnx` file the policy is saved to.
        deterministic: Whether the exported policy returns its most likely action (default) or samples one from
            external noise inputs (`noise_u` / `noise_eps`).
    """
    assert str(file_path).endswith(".onnx"), "File path must end with .onnx"

    onnx_policy = _build_onnx_policy(policy, deterministic)
    onnx_policy.train(False)  # evaluation mode
    input_names = onnx_policy.input_names()
    output_names = onnx_policy.output_names()
    torch.onnx.export(
        onnx_policy,
        onnx_policy.dummy_inputs(),
        str(file_path),
        opset_version=17,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={name: {0: "batch"} for name in input_names + output_names},
    )


def _build_onnx_policy(policy: BasePolicy, deterministic: bool) -> "OnnxPolicy":
    if isinstance(policy, ActorCriticPolicy):
        return ActorCriticOnnxPolicy(policy, deterministic)
    if isinstance(policy, SACPolicy):
        return SACOnnxPolicy(policy, deterministic)
    if isinstance(policy, TD3Policy):
        return TD3OnnxPolicy(policy, deterministic)
    if isinstance(policy, DQNPolicy):
        return DQNOnnxPolicy(policy, deterministic)
    raise NotImplementedError(f"ONNX export is not supported for policies of type {type(policy).__name__}.")


class OnnxPolicy(torch.nn.Module):
    """Wrapper of an SB3 policy with the exported inputs (observations, noise) and outputs (see module docstring)."""

    def __init__(self, policy: BasePolicy, deterministic: bool):
        super().__init__()
        self.policy = policy
        self.deterministic = deterministic
        self.observation_space = policy.observation_space
        self.action_space = policy.action_space
        self.is_dict_observation = isinstance(self.observation_space, gym.spaces.Dict)
        self.observation_names = list(self.observation_space.spaces.keys()) if self.is_dict_observation else ["obs"]
        if isinstance(self.action_space, gym.spaces.Box):
            self.register_buffer("action_low", torch.as_tensor(self.action_space.low, dtype=torch.float32))
            self.register_buffer("action_high", torch.as_tensor(self.action_space.high, dtype=torch.float32))

    # --- contract ---

    def output_names(self) -> List[str]:
        raise NotImplementedError

    def noise_name(self) -> str:
        return "noise_eps" if isinstance(self.action_space, gym.spaces.Box) else "noise_u"

    def noise_dim(self) -> int:
        raise NotImplementedError

    def input_names(self) -> List[str]:
        return self.observation_names + ([] if self.deterministic else [self.noise_name()])

    def dummy_inputs(self) -> Tuple[torch.Tensor, ...]:
        device = self.policy.device
        spaces = (
            [self.observation_space.spaces[name] for name in self.observation_names]
            if self.is_dict_observation
            else [self.observation_space]
        )
        observations = tuple(torch.zeros((1, *space.shape), dtype=torch.float32, device=device) for space in spaces)
        if self.deterministic:
            return observations
        return observations + (torch.zeros((1, self.noise_dim()), dtype=torch.float32, device=device),)

    # --- forward ---

    def forward(self, *inputs: torch.Tensor):
        observations = inputs[: len(self.observation_names)]
        noise = None if self.deterministic else inputs[-1]
        observation = dict(zip(self.observation_names, observations)) if self.is_dict_observation else observations[0]
        return self.forward_policy(observation, noise)

    def forward_policy(self, observation, noise: Optional[torch.Tensor]):
        raise NotImplementedError

    # --- shared action helpers ---

    def clip_action(self, action: torch.Tensor) -> torch.Tensor:
        """Clip Box actions to the action space (SB3 `predict` for policies without squashing)."""
        return torch.max(torch.min(action, self.action_high), self.action_low)

    def unscale_action(self, scaled_action: torch.Tensor) -> torch.Tensor:
        """Rescale actions from [-1, 1] to the action space (SB3 `predict` for squashing policies)."""
        return self.action_low + 0.5 * (scaled_action + 1.0) * (self.action_high - self.action_low)

    def reshape_box_action(self, action: torch.Tensor) -> torch.Tensor:
        return action.reshape((-1, *self.action_space.shape))


def categorical_actions_and_log_prob(
    logits: torch.Tensor, head_dims: List[int], noise_u: Optional[torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Actions (int64, one column per head) and their summed log-probability for (multi-)categorical logits.
    Without noise the most likely action of each head is taken, otherwise one is sampled with the Gumbel-max trick.
    """
    if noise_u is not None:
        uniform = torch.clamp(noise_u, GUMBEL_EPSILON, 1.0 - GUMBEL_EPSILON)
        gumbel_heads = torch.split(-torch.log(-torch.log(uniform)), head_dims, dim=1)
    actions, log_prob = [], 0.0
    for index, head_logits in enumerate(torch.split(logits, head_dims, dim=1)):
        sampling_logits = head_logits if noise_u is None else head_logits + gumbel_heads[index]
        head_action = sampling_logits.argmax(dim=1, keepdim=True)
        # Positive gather axis (ONNX' reference evaluator fails on negative GatherElements axes)
        log_prob = log_prob + torch.log_softmax(head_logits, dim=1).gather(1, head_action).squeeze(1)
        actions.append(head_action)
    return torch.cat(actions, dim=1), log_prob


def gaussian_log_prob(action: torch.Tensor, mean: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
    """Log-probability of a diagonal Gaussian, summed over the action dimensions (batch,)."""
    log_probabilities = -((action - mean) ** 2) / (2 * torch.exp(2 * log_std)) - log_std - 0.5 * math.log(2 * math.pi)
    return log_probabilities.sum(dim=1)


class ActorCriticOnnxPolicy(OnnxPolicy):
    """PPO, A2C, TRPO (`ActorCriticPolicy`): outputs `action`, `values`, `log_prob`."""

    def __init__(self, policy: ActorCriticPolicy, deterministic: bool):
        super().__init__(policy, deterministic)
        if getattr(policy, "use_sde", False):
            raise NotImplementedError("ONNX export of policies with gSDE (use_sde=True) is not supported.")
        if hasattr(policy, "lstm_actor"):
            raise NotImplementedError("ONNX export of recurrent policies is not supported.")
        if isinstance(policy.action_dist, CategoricalDistribution):
            self.head_dims = [int(self.action_space.n)]
        elif isinstance(policy.action_dist, MultiCategoricalDistribution):
            self.head_dims = [int(n) for n in self.action_space.nvec]
        elif isinstance(policy.action_dist, DiagGaussianDistribution):
            self.head_dims = None
        else:
            raise NotImplementedError(
                f"ONNX export is not supported for the action distribution {type(policy.action_dist).__name__}."
            )

    def output_names(self) -> List[str]:
        return ["action", "values", "log_prob"]

    def noise_dim(self) -> int:
        return sum(self.head_dims) if self.head_dims else int(np.prod(self.action_space.shape))

    def forward_policy(self, observation, noise: Optional[torch.Tensor]):
        policy = self.policy
        # Like `ActorCriticPolicy.forward`
        features = policy.extract_features(observation)
        if policy.share_features_extractor:
            latent_pi, latent_vf = policy.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = policy.mlp_extractor.forward_actor(pi_features)
            latent_vf = policy.mlp_extractor.forward_critic(vf_features)
        values = policy.value_net(latent_vf)
        action_output = policy.action_net(latent_pi)

        if self.head_dims:
            action, log_prob = categorical_actions_and_log_prob(action_output, self.head_dims, noise)
            return action, values, log_prob

        mean, log_std = action_output, policy.log_std.expand_as(action_output)
        action = mean if noise is None else mean + torch.exp(log_std) * noise
        log_prob = gaussian_log_prob(action, mean, log_std)
        action = self.reshape_box_action(action)
        action = self.unscale_action(action) if policy.squash_output else self.clip_action(action)
        return action, values, log_prob


class SACOnnxPolicy(OnnxPolicy):
    """SAC (`SACPolicy`): outputs `action` and `log_prob` (of the squashed action, like SB3's `action_log_prob`)."""

    def __init__(self, policy: SACPolicy, deterministic: bool):
        super().__init__(policy, deterministic)
        if policy.actor.use_sde:
            raise NotImplementedError("ONNX export of policies with gSDE (use_sde=True) is not supported.")

    def output_names(self) -> List[str]:
        return ["action", "log_prob"]

    def noise_dim(self) -> int:
        return int(np.prod(self.action_space.shape))

    def forward_policy(self, observation, noise: Optional[torch.Tensor]):
        mean, log_std, _ = self.policy.actor.get_action_dist_params(observation)
        gaussian_action = mean if noise is None else mean + torch.exp(log_std) * noise
        squashed_action = torch.tanh(gaussian_action)
        log_prob = gaussian_log_prob(gaussian_action, mean, log_std) - torch.log(
            1 - squashed_action**2 + SQUASH_EPSILON
        ).sum(dim=1)
        return self.unscale_action(self.reshape_box_action(squashed_action)), log_prob


class TD3OnnxPolicy(OnnxPolicy):
    """TD3, DDPG (`TD3Policy`): outputs `action`. The exploration noise is not part of the policy."""

    def __init__(self, policy: TD3Policy, deterministic: bool):
        if not deterministic:
            raise ValueError(
                "TD3/DDPG policies are deterministic (their exploration noise is added by the algorithm); "
                "export them with deterministic=True."
            )
        super().__init__(policy, deterministic)

    def output_names(self) -> List[str]:
        return ["action"]

    def forward_policy(self, observation, noise: Optional[torch.Tensor]):
        return self.unscale_action(self.reshape_box_action(self.policy.actor(observation)))


class DQNOnnxPolicy(OnnxPolicy):
    """DQN (`DQNPolicy`): outputs the greedy `action` (batch, 1) and the `q_values`."""

    def __init__(self, policy: DQNPolicy, deterministic: bool):
        if not deterministic:
            raise ValueError(
                "DQN policies are greedy (epsilon-greedy exploration is added by the algorithm); "
                "export them with deterministic=True."
            )
        super().__init__(policy, deterministic)

    def output_names(self) -> List[str]:
        return ["action", "q_values"]

    def forward_policy(self, observation, noise: Optional[torch.Tensor]):
        q_values = self.policy.q_net(observation)
        return q_values.argmax(dim=1, keepdim=True), q_values

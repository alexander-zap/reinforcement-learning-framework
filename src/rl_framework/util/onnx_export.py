"""
ONNX export of Stable-Baselines3 (and SB3-contrib) policies, used by the SB3 and imitation agents.

The exported contract follows hfm-inference-server's flat ONNX wrappers (`onnx_export/sb3.py`), extended by the
values and log-probabilities of actor-critic policies:

- Inputs (all float32, dynamic batch axis `batch`), in this order:
    - `obs`: the observation; for Dict observation spaces one input per key, named like the key.
      Discrete/MultiDiscrete/image observations are fed as float too (the graph preprocesses them like SB3).
      Observations are laid out like the policy's observation space, e.g., images channel-first (SB3 transposes
      channel-last images in `predict` and with `VecTransposeImage` in training).
    - MaskablePPO only: `action_masks` (batch, number of action logits), 1 for allowed and 0 for masked actions,
      like SB3-contrib's `action_masks` (all ones: no masking).
    - Stochastic export only: external noise instead of random number nodes, so that sampling is reproducible:
      `noise_u` (Uniform(0, 1)) for discrete action spaces, one value per action logit (Gumbel-max sampling) or per
      binary action (MultiBinary: action 1 if the noise is below its probability);
      `noise_eps` (Normal(0, 1), one value per action dimension; reparameterization) for Box action spaces.
- Outputs (dynamic batch axis `batch`):
    - `action`: Discrete/MultiDiscrete/MultiBinary actions as int64 of shape (batch, number of action heads), i.e.
      (batch, 1) for Discrete; Box actions postprocessed like SB3's `predict` (clipped to the action space, or
      unscaled for squashing policies).
    - Actor-critic policies (PPO, A2C, TRPO, MaskablePPO): `values` (batch, 1) and `log_prob` (batch,) of the
      (unpostprocessed) action, like SB3's `policy.forward`.
    - SAC, TQC, CrossQ: `log_prob` (batch,) of the squashed action.
    - DQN, QR-DQN: `q_values` (batch, number of actions); for QR-DQN the mean over the quantiles.
- The graph includes the observation preprocessing and the features extractor, but not the agent's
  `FeaturesExtractor.preprocess` (applied to the environment's observations before the policy).
"""

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
from sb3_contrib.common.maskable.distributions import (
    MaskableBernoulliDistribution,
    MaskableCategoricalDistribution,
    MaskableMultiCategoricalDistribution,
)
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.crossq.policies import CrossQPolicy
from sb3_contrib.qrdqn.policies import QRDQNPolicy
from sb3_contrib.tqc.policies import TQCPolicy
from stable_baselines3.common.distributions import (
    BernoulliDistribution,
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
# Logit of masked actions (like SB3-contrib's `MaskableCategorical`)
MASKED_LOGIT = -1e8


def export_sb3_policy_as_onnx(policy: BasePolicy, file_path: Path, deterministic: bool = True) -> None:
    """
    Export an SB3 policy to ONNX (see the module docstring for the exported inputs and outputs).

    Supported policies:
        - actor-critic policies (PPO, A2C, TRPO, MaskablePPO) with Discrete, MultiDiscrete, MultiBinary or Box action
          spaces (without gSDE),
        - SAC, TQC, CrossQ (without gSDE),
        - TD3/DDPG, DQN and QR-DQN (deterministic only: their exploration is added by the algorithm).
    Recurrent policies (RecurrentPPO) and ARS are not supported.
    The policy's training mode is restored after the export.

    Args:
        policy: SB3 policy to export.
        file_path: The `.onnx` file the policy is saved to.
        deterministic: Whether the exported policy returns its most likely action (default) or samples one from
            external noise inputs (`noise_u` / `noise_eps`).
    """
    assert str(file_path).endswith(".onnx"), "File path must end with .onnx"

    onnx_policy = _build_onnx_policy(policy, deterministic)
    input_names = onnx_policy.input_names()
    output_names = onnx_policy.output_names()
    policy_was_training = policy.training
    # Evaluation mode (e.g., for the BatchRenorm layers of CrossQ), restored afterward
    policy.set_training_mode(False)
    try:
        torch.onnx.export(
            onnx_policy,
            onnx_policy.dummy_inputs(),
            str(file_path),
            opset_version=17,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes={name: {0: "batch"} for name in input_names + output_names},
        )
    finally:
        policy.set_training_mode(policy_was_training)


def _build_onnx_policy(policy: BasePolicy, deterministic: bool) -> "OnnxPolicy":
    if isinstance(policy, (ActorCriticPolicy, MaskableActorCriticPolicy)):
        return ActorCriticOnnxPolicy(policy, deterministic)
    if isinstance(policy, (SACPolicy, TQCPolicy, CrossQPolicy)):
        return SquashedGaussianActorOnnxPolicy(policy, deterministic)
    if isinstance(policy, TD3Policy):
        return TD3OnnxPolicy(policy, deterministic)
    if isinstance(policy, (DQNPolicy, QRDQNPolicy)):
        return QValueOnnxPolicy(policy, deterministic)
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

    def extra_input_dims(self) -> Dict[str, int]:
        """Inputs between the observations and the noise (name: size per batch entry)."""
        return {}

    def noise_name(self) -> str:
        return "noise_eps" if isinstance(self.action_space, gym.spaces.Box) else "noise_u"

    def noise_dim(self) -> int:
        raise NotImplementedError

    def input_names(self) -> List[str]:
        noise_names = [] if self.deterministic else [self.noise_name()]
        return self.observation_names + list(self.extra_input_dims()) + noise_names

    def dummy_inputs(self) -> Tuple[torch.Tensor, ...]:
        device = self.policy.device
        spaces = (
            [self.observation_space.spaces[name] for name in self.observation_names]
            if self.is_dict_observation
            else [self.observation_space]
        )
        inputs = [torch.zeros((1, *space.shape), dtype=torch.float32, device=device) for space in spaces]
        inputs += [
            torch.ones((1, size), dtype=torch.float32, device=device) for size in self.extra_input_dims().values()
        ]
        if not self.deterministic:
            inputs.append(torch.zeros((1, self.noise_dim()), dtype=torch.float32, device=device))
        return tuple(inputs)

    # --- forward ---

    def forward(self, *inputs: torch.Tensor):
        observations = inputs[: len(self.observation_names)]
        extra_names = list(self.extra_input_dims())
        extras = dict(
            zip(extra_names, inputs[len(self.observation_names) : len(self.observation_names) + len(extra_names)])
        )
        noise = None if self.deterministic else inputs[-1]
        observation = dict(zip(self.observation_names, observations)) if self.is_dict_observation else observations[0]
        return self.forward_policy(observation, extras, noise)

    def forward_policy(self, observation, extras: Dict[str, torch.Tensor], noise: Optional[torch.Tensor]):
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


def bernoulli_actions_and_log_prob(
    logits: torch.Tensor, noise_u: Optional[torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Binary actions (int64, one column per action) and their summed log-probability for Bernoulli logits.
    Without noise an action is 1 if its probability exceeds 0.5 (SB3's mode), otherwise 1 if the noise is below it.
    """
    probabilities = torch.sigmoid(logits)
    actions = (probabilities > 0.5) if noise_u is None else (noise_u < probabilities)
    actions = actions.to(torch.int64)
    float_actions = actions.to(torch.float32)
    log_prob = float_actions * torch.nn.functional.logsigmoid(logits) + (
        1 - float_actions
    ) * torch.nn.functional.logsigmoid(-logits)
    return actions, log_prob.sum(dim=1)


def gaussian_log_prob(action: torch.Tensor, mean: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
    """Log-probability of a diagonal Gaussian, summed over the action dimensions (batch,)."""
    log_probabilities = -((action - mean) ** 2) / (2 * torch.exp(2 * log_std)) - log_std - 0.5 * math.log(2 * math.pi)
    return log_probabilities.sum(dim=1)


class ActorCriticOnnxPolicy(OnnxPolicy):
    """PPO, A2C, TRPO (`ActorCriticPolicy`) and MaskablePPO: outputs `action`, `values`, `log_prob`."""

    def __init__(self, policy: BasePolicy, deterministic: bool):
        super().__init__(policy, deterministic)
        if getattr(policy, "use_sde", False):
            raise NotImplementedError("ONNX export of policies with gSDE (use_sde=True) is not supported.")
        if hasattr(policy, "lstm_actor"):
            raise NotImplementedError("ONNX export of recurrent policies is not supported.")
        self.is_maskable = isinstance(policy, MaskableActorCriticPolicy)
        action_distribution = policy.action_dist
        self.head_dims: Optional[List[int]] = None
        self.is_bernoulli = False
        # Maskable MultiBinary actions are categorical heads with two choices each (MaskableBernoulliDistribution)
        if isinstance(action_distribution, MaskableBernoulliDistribution):
            self.head_dims = [2] * int(np.prod(self.action_space.shape))
        elif isinstance(action_distribution, (CategoricalDistribution, MaskableCategoricalDistribution)):
            self.head_dims = [int(self.action_space.n)]
        elif isinstance(action_distribution, (MultiCategoricalDistribution, MaskableMultiCategoricalDistribution)):
            self.head_dims = [int(n) for n in self.action_space.nvec]
        elif isinstance(action_distribution, BernoulliDistribution):
            self.is_bernoulli = True
        elif not isinstance(action_distribution, DiagGaussianDistribution):
            raise NotImplementedError(
                f"ONNX export is not supported for the action distribution {type(action_distribution).__name__}."
            )

    def output_names(self) -> List[str]:
        return ["action", "values", "log_prob"]

    def extra_input_dims(self) -> Dict[str, int]:
        return {"action_masks": sum(self.head_dims)} if self.is_maskable else {}

    def noise_dim(self) -> int:
        return sum(self.head_dims) if self.head_dims else int(np.prod(self.action_space.shape))

    def forward_policy(self, observation, extras: Dict[str, torch.Tensor], noise: Optional[torch.Tensor]):
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
            if self.is_maskable:
                masked_logit = torch.full_like(action_output, MASKED_LOGIT)
                action_output = torch.where(extras["action_masks"] > 0.5, action_output, masked_logit)
            action, log_prob = categorical_actions_and_log_prob(action_output, self.head_dims, noise)
            return action, values, log_prob

        if self.is_bernoulli:
            action, log_prob = bernoulli_actions_and_log_prob(action_output, noise)
            return action, values, log_prob

        mean, log_std = action_output, policy.log_std.expand_as(action_output)
        action = mean if noise is None else mean + torch.exp(log_std) * noise
        log_prob = gaussian_log_prob(action, mean, log_std)
        action = self.reshape_box_action(action)
        action = self.unscale_action(action) if policy.squash_output else self.clip_action(action)
        return action, values, log_prob


class SquashedGaussianActorOnnxPolicy(OnnxPolicy):
    """
    SAC, TQC, CrossQ (squashed Gaussian actor): outputs `action` and `log_prob` (of the squashed action, like the
    actor's `action_log_prob`).
    """

    def __init__(self, policy: BasePolicy, deterministic: bool):
        super().__init__(policy, deterministic)
        if policy.actor.use_sde:
            raise NotImplementedError("ONNX export of policies with gSDE (use_sde=True) is not supported.")

    def output_names(self) -> List[str]:
        return ["action", "log_prob"]

    def noise_dim(self) -> int:
        return int(np.prod(self.action_space.shape))

    def forward_policy(self, observation, extras: Dict[str, torch.Tensor], noise: Optional[torch.Tensor]):
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

    def forward_policy(self, observation, extras: Dict[str, torch.Tensor], noise: Optional[torch.Tensor]):
        return self.unscale_action(self.reshape_box_action(self.policy.actor(observation)))


class QValueOnnxPolicy(OnnxPolicy):
    """DQN (`DQNPolicy`), QR-DQN (`QRDQNPolicy`): outputs the greedy `action` (batch, 1) and the `q_values`."""

    def __init__(self, policy: BasePolicy, deterministic: bool):
        if not deterministic:
            raise ValueError(
                "DQN/QR-DQN policies are greedy (epsilon-greedy exploration is added by the algorithm); "
                "export them with deterministic=True."
            )
        super().__init__(policy, deterministic)

    def output_names(self) -> List[str]:
        return ["action", "q_values"]

    def forward_policy(self, observation, extras: Dict[str, torch.Tensor], noise: Optional[torch.Tensor]):
        if isinstance(self.policy, QRDQNPolicy):
            # Quantiles (batch, quantiles, actions): the Q-values are their mean (like `QuantileNetwork._predict`)
            q_values = self.policy.quantile_net(observation).mean(dim=1)
        else:
            q_values = self.policy.q_net(observation)
        return q_values.argmax(dim=1, keepdim=True), q_values

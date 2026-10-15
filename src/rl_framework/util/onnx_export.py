"""
ONNX export of Stable-Baselines3 (and SB3-contrib) policies, used by the SB3 and imitation agents, and of d3rlpy's
algorithms (Q-learning algorithms: `D3RLPYOnnxPolicy`; decision transformers, with further inputs:
`D3RLPYTransformerOnnxPolicy`; see `docs/onnx-export.md`).

The exported contract (also described in `docs/onnx-export.md`):

- Inputs (all float32, dynamic batch axis `batch`), in this order:
    - `obs`: the observation; for Dict observation spaces one input per key, named like the key.
      Discrete/MultiDiscrete/image observations are fed as float too (the graph preprocesses them like SB3).
      Observations are laid out like the policy's observation space, e.g., images channel-first (SB3 transposes
      channel-last images in `predict` and with `VecTransposeImage` in training).
    - MaskablePPO only: `action_masks` (batch, number of action logits), 1 for allowed and 0 for masked actions,
      like SB3-contrib's `action_masks` (all ones: no masking).
    - RecurrentPPO only: `episode_starts` (batch,), 1 at the first step of an episode (resets the LSTM states), and
      the LSTM states `lstm_actor_h`, `lstm_actor_c` and, with a separate critic LSTM (`enable_critic_lstm`),
      `lstm_critic_h`, `lstm_critic_c`, each (n_lstm_layers, batch, lstm_hidden_size), i.e. batch axis 1.
      Zeros at the start; afterward the `*_out` outputs of the previous step.
    - Stochastic export only: external noise instead of random number nodes, so that sampling is reproducible:
      `noise_u` (Uniform(0, 1)) for discrete action spaces, one value per action logit (Gumbel-max sampling) or per
      binary action (MultiBinary: action 1 if the noise is below its probability);
      `noise_eps` (Normal(0, 1), one value per action dimension; reparameterization) for Box action spaces. For
      policies trained with gSDE, the action is sampled from gSDE's per-step distribution (its state-dependent
      standard deviation), i.e. like gSDE resampling its exploration matrix at every step.
- Outputs (dynamic batch axis `batch`):
    - `action`: Discrete/MultiDiscrete/MultiBinary actions as int64 of shape (batch, number of action heads), i.e.
      (batch, 1) for Discrete; Box actions postprocessed like SB3's `predict` (clipped to the action space, or
      unscaled for squashing policies).
    - Actor-critic policies (PPO, A2C, TRPO, MaskablePPO, RecurrentPPO): `values` (batch, 1) and `log_prob` (batch,)
      of the (unpostprocessed) action, like SB3's `policy.forward`.
    - RecurrentPPO: the new LSTM states `lstm_actor_h_out`, `lstm_actor_c_out` (and `lstm_critic_h_out`,
      `lstm_critic_c_out`), batch axis 1.
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
from d3rlpy import ActionSpace
from d3rlpy.algos import (
    DiscreteRandomPolicy,
    QLearningAlgoBase,
    RandomPolicy,
    TransformerAlgoBase,
)
from d3rlpy.algos.qlearning.torch.utility import DiscreteQFunctionMixin
from d3rlpy.base import LearnableBase
from d3rlpy.torch_utility import asdict_without_copy
from sb3_contrib.ars.policies import ARSPolicy
from sb3_contrib.common.maskable.distributions import (
    MaskableBernoulliDistribution,
    MaskableCategoricalDistribution,
    MaskableMultiCategoricalDistribution,
)
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from sb3_contrib.crossq.policies import CrossQPolicy
from sb3_contrib.qrdqn.policies import QRDQNPolicy
from sb3_contrib.tqc.policies import TQCPolicy
from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
)
from stable_baselines3.common.policies import ActorCriticPolicy, BasePolicy
from stable_baselines3.dqn.policies import DQNPolicy
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.td3.policies import TD3Policy

# Clamping of the uniform noise for the Gumbel-max trick (keeps the logarithms finite)
GUMBEL_EPSILON = 1e-6
# Epsilon of SB3's squashed Gaussian log-probability (`SquashedDiagGaussianDistribution.log_prob`)
SQUASH_EPSILON = 1e-6
# Logit of masked actions (like SB3-contrib's `MaskableCategorical`)
MASKED_LOGIT = -1e8


def export_sb3_policy_as_onnx(policy: BasePolicy, file_path: Path, deterministic: bool = True) -> None:
    """
    Export an SB3 policy to ONNX (see the module docstring for the exported inputs and outputs).

    Supported policies:
        - actor-critic policies (PPO, A2C, TRPO, MaskablePPO, RecurrentPPO) with Discrete, MultiDiscrete, MultiBinary
          or Box action spaces (also trained with gSDE),
        - SAC, TQC, CrossQ (also trained with gSDE),
        - TD3/DDPG, DQN, QR-DQN and ARS (deterministic only: their exploration is added by the algorithm).
    The policy's training mode is restored after the export.

    Args:
        policy: SB3 policy to export.
        file_path: The `.onnx` file the policy is saved to.
        deterministic: Whether the exported policy returns its most likely action (default) or samples one from
            external noise inputs (`noise_u` / `noise_eps`).
    """
    assert str(file_path).endswith(".onnx"), "File path must end with .onnx"

    onnx_policy = _build_onnx_policy(policy, deterministic)
    policy_was_training = policy.training
    # Evaluation mode (e.g., for the BatchRenorm layers of CrossQ), restored afterward
    policy.set_training_mode(False)
    try:
        _export(onnx_policy, file_path)
    finally:
        policy.set_training_mode(policy_was_training)


def export_d3rlpy_algorithm_as_onnx(algorithm: LearnableBase, file_path: Path, deterministic: bool = True) -> None:
    """
    Export the policy of a trained d3rlpy algorithm to ONNX, in the contract of the SB3 export (see the module
    docstring):
        - Q-learning algorithms (`D3RLPYOnnxPolicy`): the greedy policy, input `obs`, outputs `action` (like the
          algorithm's `predict`, with its observation and action scalers) and, for algorithms with a discrete
          Q-function, `q_values`.
        - Decision transformers (`D3RLPYTransformerOnnxPolicy`): one step per call, with the further inputs `reward`
          and `target_return` and the context window as inputs and outputs.
    The modules are left in evaluation mode, like after d3rlpy's `predict`.

    Args:
        algorithm: Trained (or built) d3rlpy algorithm.
        file_path: The `.onnx` file the policy is saved to.
        deterministic: Whether the most likely action is exported (default) or, for discrete decision transformers
            only, one sampled from the noise input `noise_u`. Other algorithms raise a `ValueError` for False.
    """
    assert str(file_path).endswith(".onnx"), "File path must end with .onnx"

    onnx_policy = (
        D3RLPYTransformerOnnxPolicy(algorithm, deterministic)
        if isinstance(algorithm, TransformerAlgoBase)
        else D3RLPYOnnxPolicy(algorithm, deterministic)
    )
    algorithm.impl.modules.set_eval()
    _export(onnx_policy, file_path)


def _export(onnx_policy: "OnnxPolicy", file_path: Path) -> None:
    input_names = onnx_policy.input_names()
    output_names = onnx_policy.output_names()
    torch.onnx.export(
        onnx_policy,
        onnx_policy.dummy_inputs(),
        str(file_path),
        opset_version=17,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={name: {onnx_policy.batch_axis(name): "batch"} for name in input_names + output_names},
        # TorchScript exporter: the dynamo exporter (default since torch 2.9) cannot map `dynamic_axes` to the
        # variadic `forward(*inputs)`
        dynamo=False,
    )


def _build_onnx_policy(policy: BasePolicy, deterministic: bool) -> "OnnxPolicy":
    if isinstance(policy, RecurrentActorCriticPolicy):
        return RecurrentActorCriticOnnxPolicy(policy, deterministic)
    if isinstance(policy, (ActorCriticPolicy, MaskableActorCriticPolicy)):
        return ActorCriticOnnxPolicy(policy, deterministic)
    if isinstance(policy, (SACPolicy, TQCPolicy, CrossQPolicy)):
        return SquashedGaussianActorOnnxPolicy(policy, deterministic)
    if isinstance(policy, TD3Policy):
        return TD3OnnxPolicy(policy, deterministic)
    if isinstance(policy, (DQNPolicy, QRDQNPolicy)):
        return QValueOnnxPolicy(policy, deterministic)
    if isinstance(policy, ARSPolicy):
        return ARSOnnxPolicy(policy, deterministic)
    raise NotImplementedError(f"ONNX export is not supported for policies of type {type(policy).__name__}.")


class OnnxPolicy(torch.nn.Module):
    """
    Wrapper of a policy with the exported inputs (observations, noise) and outputs (see module docstring). The spaces
    are the SB3 policy's unless given.
    """

    def __init__(
        self,
        policy,
        deterministic: bool,
        observation_space: Optional[gym.Space] = None,
        action_space: Optional[gym.Space] = None,
    ):
        super().__init__()
        self.policy = policy
        self.deterministic = deterministic
        self.observation_space = observation_space or policy.observation_space
        self.action_space = action_space or policy.action_space
        self.is_dict_observation = isinstance(self.observation_space, gym.spaces.Dict)
        self.observation_names = list(self.observation_space.spaces.keys()) if self.is_dict_observation else ["obs"]
        if isinstance(self.action_space, gym.spaces.Box):
            self.register_buffer("action_low", torch.as_tensor(self.action_space.low, dtype=torch.float32))
            self.register_buffer("action_high", torch.as_tensor(self.action_space.high, dtype=torch.float32))

    # --- contract ---

    def output_names(self) -> List[str]:
        raise NotImplementedError

    def extra_inputs(self) -> Dict[str, Tuple[Tuple[int, ...], float]]:
        """
        Inputs between the observations and the noise: name -> (shape for a batch of 1, value of the dummy input).
        """
        return {}

    def batch_axis(self, name: str) -> int:
        return 0

    def noise_name(self) -> str:
        return "noise_eps" if isinstance(self.action_space, gym.spaces.Box) else "noise_u"

    def noise_dim(self) -> int:
        raise NotImplementedError

    def input_names(self) -> List[str]:
        noise_names = [] if self.deterministic else [self.noise_name()]
        return self.observation_names + list(self.extra_inputs()) + noise_names

    def device(self):
        return self.policy.device

    def dummy_inputs(self) -> Tuple[torch.Tensor, ...]:
        device = self.device()
        spaces = (
            [self.observation_space.spaces[name] for name in self.observation_names]
            if self.is_dict_observation
            else [self.observation_space]
        )
        inputs = [torch.zeros((1, *space.shape), dtype=torch.float32, device=device) for space in spaces]
        inputs += [
            torch.full(shape, value, dtype=torch.float32, device=device)
            for shape, value in self.extra_inputs().values()
        ]
        if not self.deterministic:
            inputs.append(torch.zeros((1, self.noise_dim()), dtype=torch.float32, device=device))
        return tuple(inputs)

    # --- forward ---

    def forward(self, *inputs: torch.Tensor):
        n_observations, extra_names = len(self.observation_names), list(self.extra_inputs())
        observations = inputs[:n_observations]
        extras = dict(zip(extra_names, inputs[n_observations : n_observations + len(extra_names)]))
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

    def postprocess_box_action(self, action: torch.Tensor) -> torch.Tensor:
        """Shape of the action space, then unscaled (squashing policies) or clipped, like SB3's `predict`."""
        action = action.reshape((-1, *self.action_space.shape))
        return self.unscale_action(action) if self.policy.squash_output else self.clip_action(action)


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
        # Positive gather axis (keeps the graph evaluable by onnx' reference evaluator, which fails on negative axes)
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


def gaussian_actions_and_log_prob(
    mean: torch.Tensor, std: torch.Tensor, noise_eps: Optional[torch.Tensor], squash: bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Gaussian actions (the mean without noise, otherwise `mean + std * noise_eps`) and their log-probability, summed
    over the action dimensions. With `squash`, the actions are squashed with tanh and the log-probability corrected
    accordingly (like SB3's squashed Gaussian distributions).
    """
    action = mean if noise_eps is None else mean + std * noise_eps
    log_prob = (-((action - mean) ** 2) / (2 * std**2) - torch.log(std) - 0.5 * math.log(2 * math.pi)).sum(dim=1)
    if squash:
        action = torch.tanh(action)
        log_prob = log_prob - torch.log(1 - action**2 + SQUASH_EPSILON).sum(dim=1)
    return action, log_prob


def gsde_std(action_distribution: StateDependentNoiseDistribution, log_std: torch.Tensor, latent_sde: torch.Tensor):
    """
    Standard deviation of gSDE's per-step action distribution: the noise `latent_sde @ exploration_matrix` with
    `exploration_matrix ~ N(0, std^2)` has the variance `latent_sde^2 @ std^2` (like
    `StateDependentNoiseDistribution.proba_distribution`).
    """
    variance = torch.mm(latent_sde**2, action_distribution.get_std(log_std) ** 2)
    return torch.sqrt(variance + action_distribution.epsilon)


class ActorCriticOnnxPolicy(OnnxPolicy):
    """PPO, A2C, TRPO (`ActorCriticPolicy`) and MaskablePPO: outputs `action`, `values`, `log_prob`."""

    def __init__(self, policy: BasePolicy, deterministic: bool):
        super().__init__(policy, deterministic)
        self.is_maskable = isinstance(policy, MaskableActorCriticPolicy)
        action_distribution = policy.action_dist
        self.head_dims: Optional[List[int]] = None
        self.is_bernoulli = False
        self.is_gsde = isinstance(action_distribution, StateDependentNoiseDistribution)
        # Maskable MultiBinary actions are categorical heads with two choices each (MaskableBernoulliDistribution)
        if isinstance(action_distribution, MaskableBernoulliDistribution):
            self.head_dims = [2] * int(np.prod(self.action_space.shape))
        elif isinstance(action_distribution, (CategoricalDistribution, MaskableCategoricalDistribution)):
            self.head_dims = [int(self.action_space.n)]
        elif isinstance(action_distribution, (MultiCategoricalDistribution, MaskableMultiCategoricalDistribution)):
            self.head_dims = [int(n) for n in self.action_space.nvec]
        elif isinstance(action_distribution, BernoulliDistribution):
            self.is_bernoulli = True
        elif not isinstance(action_distribution, (DiagGaussianDistribution, StateDependentNoiseDistribution)):
            raise NotImplementedError(
                f"ONNX export is not supported for the action distribution {type(action_distribution).__name__}."
            )

    def output_names(self) -> List[str]:
        return ["action", "values", "log_prob"]

    def extra_inputs(self) -> Dict[str, Tuple[Tuple[int, ...], float]]:
        return {"action_masks": ((1, sum(self.head_dims)), 1.0)} if self.is_maskable else {}

    def noise_dim(self) -> int:
        return sum(self.head_dims) if self.head_dims else int(np.prod(self.action_space.shape))

    def latents(self, observation, extras: Dict[str, torch.Tensor]):
        """Latent codes of the actor and critic, and further outputs (like `ActorCriticPolicy.forward`)."""
        policy = self.policy
        features = policy.extract_features(observation)
        if policy.share_features_extractor:
            latent_pi, latent_vf = policy.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = policy.mlp_extractor.forward_actor(pi_features)
            latent_vf = policy.mlp_extractor.forward_critic(vf_features)
        return latent_pi, latent_vf, ()

    def forward_policy(self, observation, extras: Dict[str, torch.Tensor], noise: Optional[torch.Tensor]):
        policy = self.policy
        latent_pi, latent_vf, further_outputs = self.latents(observation, extras)
        values = policy.value_net(latent_vf)
        action_output = policy.action_net(latent_pi)

        if self.head_dims:
            if self.is_maskable:
                masked_logit = torch.full_like(action_output, MASKED_LOGIT)
                action_output = torch.where(extras["action_masks"] > 0.5, action_output, masked_logit)
            action, log_prob = categorical_actions_and_log_prob(action_output, self.head_dims, noise)
        elif self.is_bernoulli:
            action, log_prob = bernoulli_actions_and_log_prob(action_output, noise)
        else:
            if self.is_gsde:
                std = gsde_std(policy.action_dist, policy.log_std, latent_pi)
                squash = policy.action_dist.bijector is not None
            else:
                std, squash = torch.exp(policy.log_std).expand_as(action_output), False
            action, log_prob = gaussian_actions_and_log_prob(action_output, std, noise, squash)
            action = self.postprocess_box_action(action)
        return (action, values, log_prob) + tuple(further_outputs)


class RecurrentActorCriticOnnxPolicy(ActorCriticOnnxPolicy):
    """
    RecurrentPPO (`RecurrentActorCriticPolicy`): one step per call, with `episode_starts` and the LSTM states as
    inputs and the new LSTM states as further outputs.
    """

    def __init__(self, policy: RecurrentActorCriticPolicy, deterministic: bool):
        super().__init__(policy, deterministic)
        lstm = policy.lstm_actor
        self.state_shape = (lstm.num_layers, 1, lstm.hidden_size)
        self.state_names = ["lstm_actor_h", "lstm_actor_c"]
        if policy.lstm_critic is not None:
            self.state_names += ["lstm_critic_h", "lstm_critic_c"]

    def output_names(self) -> List[str]:
        return super().output_names() + [f"{name}_out" for name in self.state_names]

    def extra_inputs(self) -> Dict[str, Tuple[Tuple[int, ...], float]]:
        return {"episode_starts": ((1,), 0.0), **{name: (self.state_shape, 0.0) for name in self.state_names}}

    def batch_axis(self, name: str) -> int:
        return 1 if name.startswith("lstm_") else 0

    @staticmethod
    def lstm_step(lstm: torch.nn.LSTM, features, hidden, cell, episode_starts):
        """
        One LSTM step, with the states reset at episode starts (like `RecurrentActorCriticPolicy._process_sequence`,
        without its branching on tensor values, which tracing would fix).
        """
        keep = (1.0 - episode_starts).view(1, -1, 1)
        output, (hidden, cell) = lstm(features.unsqueeze(0), (keep * hidden, keep * cell))
        return output.squeeze(0), hidden, cell

    def latents(self, observation, extras: Dict[str, torch.Tensor]):
        policy = self.policy
        features = policy.extract_features(observation)
        pi_features, vf_features = (features, features) if policy.share_features_extractor else features
        episode_starts = extras["episode_starts"]
        latent_pi, actor_h, actor_c = self.lstm_step(
            policy.lstm_actor, pi_features, extras["lstm_actor_h"], extras["lstm_actor_c"], episode_starts
        )
        states = [actor_h, actor_c]
        if policy.lstm_critic is not None:
            latent_vf, critic_h, critic_c = self.lstm_step(
                policy.lstm_critic, vf_features, extras["lstm_critic_h"], extras["lstm_critic_c"], episode_starts
            )
            states += [critic_h, critic_c]
        elif policy.shared_lstm:
            latent_vf = latent_pi
        else:
            latent_vf = policy.critic(vf_features)
        latent_pi = policy.mlp_extractor.forward_actor(latent_pi)
        latent_vf = policy.mlp_extractor.forward_critic(latent_vf)
        return latent_pi, latent_vf, states


class SquashedGaussianActorOnnxPolicy(OnnxPolicy):
    """
    SAC, TQC, CrossQ (squashed Gaussian actor): outputs `action` and `log_prob` (of the squashed action, like the
    actor's `action_log_prob`).
    """

    def output_names(self) -> List[str]:
        return ["action", "log_prob"]

    def noise_dim(self) -> int:
        return int(np.prod(self.action_space.shape))

    def forward_policy(self, observation, extras: Dict[str, torch.Tensor], noise: Optional[torch.Tensor]):
        actor = self.policy.actor
        mean, log_std, distribution_kwargs = actor.get_action_dist_params(observation)
        if actor.use_sde:
            std = gsde_std(actor.action_dist, log_std, distribution_kwargs["latent_sde"])
        else:
            std = torch.exp(log_std)
        action, log_prob = gaussian_actions_and_log_prob(mean, std, noise, squash=True)
        return self.postprocess_box_action(action), log_prob


class DeterministicOnnxPolicy(OnnxPolicy):
    """Policies without policy-side sampling (their exploration is added by the algorithm)."""

    def __init__(self, policy, deterministic: bool, *spaces: gym.Space):
        if not deterministic:
            raise ValueError(
                f"{type(policy).__name__} policies are deterministic (their exploration is added by the algorithm); "
                "export them with deterministic=True."
            )
        super().__init__(policy, deterministic, *spaces)


class TD3OnnxPolicy(DeterministicOnnxPolicy):
    """TD3, DDPG (`TD3Policy`): outputs `action`."""

    def output_names(self) -> List[str]:
        return ["action"]

    def forward_policy(self, observation, extras: Dict[str, torch.Tensor], noise: Optional[torch.Tensor]):
        return self.postprocess_box_action(self.policy.actor(observation))


class QValueOnnxPolicy(DeterministicOnnxPolicy):
    """DQN (`DQNPolicy`), QR-DQN (`QRDQNPolicy`): outputs the greedy `action` (batch, 1) and the `q_values`."""

    def output_names(self) -> List[str]:
        return ["action", "q_values"]

    def forward_policy(self, observation, extras: Dict[str, torch.Tensor], noise: Optional[torch.Tensor]):
        if isinstance(self.policy, QRDQNPolicy):
            # Quantiles (batch, quantiles, actions): the Q-values are their mean (like `QuantileNetwork._predict`)
            q_values = self.policy.quantile_net(observation).mean(dim=1)
        else:
            q_values = self.policy.q_net(observation)
        return q_values.argmax(dim=1, keepdim=True), q_values


class ARSOnnxPolicy(DeterministicOnnxPolicy):
    """ARS (`ARSPolicy`, `ARSLinearPolicy`): outputs `action` (like `ARSPolicy.forward` and `predict`)."""

    def output_names(self) -> List[str]:
        return ["action"]

    def forward_policy(self, observation, extras: Dict[str, torch.Tensor], noise: Optional[torch.Tensor]):
        policy = self.policy
        output = policy.action_net(policy.extract_features(observation, policy.features_extractor))
        if isinstance(self.action_space, gym.spaces.Box):
            return self.postprocess_box_action(output)
        return output.argmax(dim=1, keepdim=True)


class D3RLPYOnnxPolicyBase(OnnxPolicy):
    """d3rlpy algorithms: their spaces (from the observation shape and action size) and networks."""

    def __init__(self, algorithm: LearnableBase, deterministic: bool):
        if isinstance(algorithm, (RandomPolicy, DiscreteRandomPolicy)):
            raise NotImplementedError(f"ONNX export is not supported for {type(algorithm).__name__} (no networks).")
        if algorithm.impl is None:
            raise ValueError("Cannot export an uninitialized algorithm. Start a training first to initialize.")
        observation_shape = algorithm.impl.observation_shape
        if isinstance(observation_shape[0], (list, tuple)):
            raise NotImplementedError("ONNX export is not supported for d3rlpy algorithms with tuple observations.")
        observation_space = gym.spaces.Box(-np.inf, np.inf, observation_shape, np.float32)
        action_size = algorithm.impl.action_size
        self.is_discrete = algorithm.get_action_type() == ActionSpace.DISCRETE
        action_space = (
            gym.spaces.Discrete(action_size)
            if self.is_discrete
            else gym.spaces.Box(-np.inf, np.inf, (action_size,), np.float32)
        )
        super().__init__(algorithm, deterministic, observation_space, action_space)
        # Registered as submodules, so that their parameters are exported as weights (not as constants)
        self.networks = torch.nn.ModuleDict(
            {
                name: module
                for name, module in asdict_without_copy(algorithm.impl.modules).items()
                if isinstance(module, torch.nn.Module)
            }
        )

    def device(self):
        return self.policy.impl.device

    def require_deterministic(self):
        if not self.deterministic:
            raise ValueError(
                f"{type(self.policy).__name__} exports its greedy policy only; export it with deterministic=True."
            )


class D3RLPYOnnxPolicy(D3RLPYOnnxPolicyBase):
    """
    d3rlpy Q-learning algorithms (`QLearningAlgoBase`): outputs the greedy `action` like the algorithm's `predict`
    (discrete: (batch, 1); continuous: not clipped, since d3rlpy does not know the action space's bounds) and, for
    algorithms with a discrete Q-function (DQN, DoubleDQN, NFQ, DiscreteCQL, DiscreteBCQ, DiscreteSAC), `q_values`:
    their mean over the Q-function ensemble, like `predict_value`. The action is their argmax except for DiscreteBCQ
    (the best action its imitator allows) and DiscreteSAC (the most likely action of its policy).
    """

    def __init__(self, algorithm: QLearningAlgoBase, deterministic: bool):
        super().__init__(algorithm, deterministic)
        self.require_deterministic()
        self.has_q_values = isinstance(algorithm.impl, DiscreteQFunctionMixin)

    def output_names(self) -> List[str]:
        return ["action", "q_values"] if self.has_q_values else ["action"]

    def forward_policy(self, observation, extras: Dict[str, torch.Tensor], noise: Optional[torch.Tensor]):
        config, impl = self.policy.config, self.policy.impl
        if config.observation_scaler:
            observation = config.observation_scaler.transform(observation)
        action = impl.inner_predict_best_action(observation)
        if config.action_scaler:
            action = config.action_scaler.reverse_transform(action)
        if self.is_discrete:
            action = action.reshape(-1, 1)
        if not self.has_q_values:
            return action
        # Private forwarder of the Q-functions, like `DiscreteQFunctionMixin.inner_predict_value`
        return action, impl._q_func_forwarder.compute_expected_q(observation, reduction="mean")


class D3RLPYTransformerOnnxPolicy(D3RLPYOnnxPolicyBase):
    """
    d3rlpy decision transformers (`TransformerAlgoBase`), one step per call like d3rlpy's `StatefulTransformerWrapper`:
    inputs `reward` (of the previous step, 0 at the episode start), `target_return` (read at the episode start),
    `episode_starts` and the context window `dt_*` (zeros at the start, afterward the `*_out` outputs of the previous
    step); outputs `action` and the new context window. Discrete actions are the argmax of the logits, or sampled with
    the Gumbel-max trick from `noise_u` (like `SoftmaxTransformerActionSampler` with temperature 1); continuous actions
    are deterministic only (like `IdentityTransformerActionSampler`) and not clipped.
    """

    STATE_NAMES = ["dt_observations", "dt_actions", "dt_returns_to_go", "dt_timesteps", "dt_return_rest", "dt_timestep"]

    def __init__(self, algorithm: TransformerAlgoBase, deterministic: bool):
        super().__init__(algorithm, deterministic)
        if not self.is_discrete:
            self.require_deterministic()
        self.context_size = algorithm.config.context_size
        self.max_timestep = algorithm.config.max_timestep
        self.action_dim = 1 if self.is_discrete else algorithm.impl.action_size

    def output_names(self) -> List[str]:
        return ["action"] + [f"{name}_out" for name in self.STATE_NAMES]

    def extra_inputs(self) -> Dict[str, Tuple[Tuple[int, ...], float]]:
        window = (1, self.context_size)
        return {
            "reward": ((1,), 0.0),
            "target_return": ((1,), 0.0),
            "episode_starts": ((1,), 1.0),
            "dt_observations": ((*window, *self.observation_space.shape), 0.0),
            "dt_actions": ((*window, self.action_dim), 0.0),
            "dt_returns_to_go": ((*window, 1), 0.0),
            "dt_timesteps": (window, 0.0),
            "dt_return_rest": ((1,), 0.0),
            "dt_timestep": ((1,), 0.0),
        }

    def noise_dim(self) -> int:
        return int(self.action_space.n)

    def forward_policy(self, observation, extras: Dict[str, torch.Tensor], noise: Optional[torch.Tensor]):
        config = self.policy.config
        episode_starts = extras["episode_starts"]
        starts, reward = episode_starts > 0.5, extras["reward"]
        # Reset at episode starts (like `StatefulTransformerWrapper.reset`): an empty context (zeros, d3rlpy's
        #   padding), the target return as the return still to achieve, and timestep 1
        return_rest = torch.where(starts, extras["target_return"], extras["dt_return_rest"])
        timestep = torch.where(starts, torch.ones_like(reward), extras["dt_timestep"])

        def append(window, value):
            """The window shifted by one step (dropping its oldest entry), with the value appended."""
            kept_window = window * (1.0 - episode_starts).view(-1, *([1] * (window.dim() - 1)))
            return torch.cat([kept_window[:, 1:], value.unsqueeze(1)], dim=1)

        observations = append(extras["dt_observations"], observation)
        # The action of this step is not known yet: d3rlpy's pad action (zeros)
        actions = append(extras["dt_actions"], torch.zeros_like(extras["dt_actions"][:, 0]))
        returns_to_go = append(extras["dt_returns_to_go"], (return_rest - reward).unsqueeze(1))
        timesteps = append(extras["dt_timesteps"], timestep)

        # Scalers applied to the whole context, like `TorchTransformerInput.from_numpy`
        scaled_observations, scaled_actions, scaled_returns_to_go = observations, actions, returns_to_go
        if config.observation_scaler:
            scaled_observations = config.observation_scaler.transform(observations)
        if config.action_scaler:
            scaled_actions = config.action_scaler.transform(actions)
        if config.reward_scaler:
            scaled_returns_to_go = config.reward_scaler.transform(returns_to_go)
        transformer_output = self.networks["transformer"](
            scaled_observations, scaled_actions, scaled_returns_to_go, timesteps.long()
        )

        if self.is_discrete:
            # (probabilities, logits) of every position: the logits of the last one
            action, _ = categorical_actions_and_log_prob(transformer_output[1][:, -1], [self.noise_dim()], noise)
            window_action = action.to(torch.float32)
        else:
            action = transformer_output[:, -1]
            if config.action_scaler:
                action = config.action_scaler.reverse_transform(action)
            window_action = action
        actions = torch.cat([actions[:, :-1], window_action.unsqueeze(1)], dim=1)
        next_timestep = torch.clamp(timestep + 1.0, max=float(self.max_timestep))
        return action, observations, actions, returns_to_go, timesteps, return_rest - reward, next_timestep

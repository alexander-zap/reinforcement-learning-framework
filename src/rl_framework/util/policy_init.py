"""
Optional initial action bias for the policy of a freshly created SB3 model (`initial_action_bias` algorithm parameter).

The values of the biases of the policy's action output layer, in the order of its outputs:
- PPO, A2C, TRPO (the action distribution's `action_net`): for a MultiDiscrete action space, the logits of each action
  head, as one sequence per head (e.g. `[[0, 0, 1], [0, 0, 0]]` for [3, 3]) or all in one flat sequence; for Discrete,
  its n logits; for Box, an offset per dimension of the action mean.
- SAC, TD3 (the actor's mean, Box action spaces only): an offset per dimension before tanh squashing, so the initial
  mean action is about tanh(bias), scaled to the action bounds. TD3's target actor gets the same bias.
- ARS (Box action spaces only): an offset per dimension of its deterministic action.
DQN and QR-DQN are not supported (see `UNSUPPORTED_ALGORITHMS`), nor ARS with a Discrete action space (see
`apply_action_bias`).

With default orthogonal initialisation the action head outputs about 0 at first, so a head's logits `[0, 0, 1]` give
approximate initial probabilities 0.21/0.21/0.58. These probabilities describe initialisation only: network outputs
also depend on observations, and training is free to change the biases. Off-policy algorithms take random actions for
their first `learning_starts` steps, which the bias does not affect. The bias is saved with the policy; loaded models
are never reinitialised.
"""

import logging
from typing import List, Optional, Sequence, Type

import numpy as np
import torch as th
from gymnasium import spaces
from sb3_contrib import ARS, QRDQN
from stable_baselines3 import DQN
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import BasePolicy

_Q_VALUE_REASON = (
    "it acts on the argmax of its Q-values, so a bias would not shift action probabilities but make the biased action "
    "the greedy choice, until TD updates remove the false value estimate (and early epsilon-greedy exploration ignores "
    "it)"
)
# Algorithms whose policy has an output layer, but for which a bias would not shift action probabilities.
UNSUPPORTED_ALGORITHMS = {
    DQN: _Q_VALUE_REASON,
    QRDQN: _Q_VALUE_REASON,
}


def validate_initial_action_bias(
    initial_action_bias: Optional[Sequence], algorithm_class: Type[BaseAlgorithm]
) -> Optional[np.ndarray]:
    """Return initial_action_bias as a flat array (sequences per action head joined in order), or None when not set."""
    if initial_action_bias is None:
        return None
    for unsupported, reason in UNSUPPORTED_ALGORITHMS.items():
        # Named by the base class: an AsyncSB3 agent's algorithm class is an injected subclass of it.
        if issubclass(algorithm_class, unsupported):
            raise ValueError(f"initial_action_bias does not support {unsupported.__name__}: {reason}.")
    value = initial_action_bias
    sequence_types = (list, tuple, np.ndarray)
    nested = isinstance(value, sequence_types) and len(value) > 0 and all(isinstance(h, sequence_types) for h in value)
    heads = []
    for head in value if nested else [value]:
        try:
            head_bias = np.asarray(head, dtype=np.float64)
        except (TypeError, ValueError):
            head_bias = None
        if head_bias is None or head_bias.ndim != 1 or head_bias.size == 0:
            raise ValueError(
                f"initial_action_bias must be a nonempty sequence of numbers, or one per action head, got {value!r}"
            )
        heads.append(head_bias)
    bias = np.concatenate(heads)
    if not np.all(np.isfinite(bias)):
        raise ValueError(f"initial_action_bias entries must be finite numbers, got {value!r}")
    return bias


def _last_linear(module: th.nn.Module) -> th.nn.Linear:
    return [layer for layer in module.modules() if isinstance(layer, th.nn.Linear)][-1]


def _action_output_layers(policy: BasePolicy) -> List[th.nn.Linear]:
    """The layer that outputs the policy's actions, and its copy in a target network (which must match)."""
    # PPO, A2C, TRPO (and ARS): the action distribution's logits or mean.
    if hasattr(policy, "action_net"):
        return [_last_linear(policy.action_net)]
    # SAC, TD3 (and TQC, CrossQ): the actor's mean; TD3 also keeps a target actor.
    actors = [getattr(policy, name, None) for name in ("actor", "actor_target")]
    return [_last_linear(actor.mu) for actor in actors if actor is not None and hasattr(actor, "mu")]


def apply_action_bias(algorithm: BaseAlgorithm, action_bias: np.ndarray) -> None:
    """Set the biases of the algorithm's policy's action output layer (and of its target copy) to action_bias."""
    if isinstance(algorithm, ARS) and isinstance(algorithm.action_space, spaces.Discrete):
        raise ValueError(
            "initial_action_bias does not support ARS with a Discrete action space: its policy is deterministic and "
            "takes the argmax of its logits, so a bias would make the biased action its only choice."
        )
    layers = _action_output_layers(algorithm.policy)
    if not layers:
        raise ValueError(
            f"initial_action_bias needs a policy with an action output layer (PPO, A2C, TRPO, SAC, TD3), "
            f"got {type(algorithm.policy).__name__}"
        )
    for layer in layers:
        if layer.bias is None:
            raise ValueError("initial_action_bias needs an action output layer with a bias, the policy's has none")
        if len(action_bias) != layer.bias.numel():
            raise ValueError(
                f"initial_action_bias has {len(action_bias)} entries but the policy's action output layer for "
                f"{algorithm.action_space} has {layer.bias.numel()} outputs"
            )
        with th.no_grad():
            layer.bias.copy_(layer.bias.new_tensor(action_bias))
    logging.info("Initialised the policy's action bias to %s", action_bias.tolist())

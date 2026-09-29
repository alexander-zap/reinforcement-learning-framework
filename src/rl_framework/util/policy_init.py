"""
Optional initial action bias for the policy of a freshly created SB3 model (`initial_action_bias` algorithm parameter).

The values of the biases of the policy's action output layer, in the order of its outputs:
- PPO, A2C, TRPO (the action distribution's `action_net`): for a MultiDiscrete action space, the logits of each action
  head, as one sequence per head (e.g. `[[0, 0, 1], [0, 0, 0]]` for [3, 3]) or all in one flat sequence; for Discrete,
  its n logits; for Box, an offset per dimension of the action mean.
- SAC, TD3 (the actor's mean, Box action spaces only): an offset per dimension before tanh squashing, so the initial
  mean action is about tanh(bias), scaled to the action bounds. TD3's target actor gets the same bias.
DQN is not supported (see `validate_initial_action_bias`).

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
from stable_baselines3 import DQN
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import BasePolicy


def validate_initial_action_bias(
    initial_action_bias: Optional[Sequence], algorithm_class: Type[BaseAlgorithm]
) -> Optional[np.ndarray]:
    """Return initial_action_bias as a flat array (sequences per action head joined in order), or None when not set."""
    if initial_action_bias is None:
        return None
    if issubclass(algorithm_class, DQN):
        raise ValueError(
            "initial_action_bias does not support DQN: it acts on the argmax of its Q-values, so a bias would not "
            "shift action probabilities but make the biased action the greedy choice, until TD updates remove the "
            "false value estimate (and early epsilon-greedy exploration ignores it). Use PPO, A2C, TRPO, SAC or TD3."
        )
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


def _action_output_layers(policy: BasePolicy) -> List[th.nn.Linear]:
    """The layer that outputs the policy's actions, and its copy in a target network (which must match)."""
    # PPO, A2C, TRPO: the action distribution's logits or mean.
    if hasattr(policy, "action_net"):
        return [policy.action_net]
    # SAC and TD3: the last linear layer of the actor's mean; TD3 also keeps a target actor.
    actors = [getattr(policy, name, None) for name in ("actor", "actor_target")]
    return [
        [module for module in actor.mu.modules() if isinstance(module, th.nn.Linear)][-1]
        for actor in actors
        if actor is not None and hasattr(actor, "mu")
    ]


def apply_action_bias(algorithm: BaseAlgorithm, action_bias: np.ndarray) -> None:
    """Set the biases of the algorithm's policy's action output layer (and of its target copy) to action_bias."""
    layers = _action_output_layers(algorithm.policy)
    if not layers:
        raise ValueError(
            f"initial_action_bias needs a policy with an action output layer (PPO, A2C, TRPO, SAC, TD3), "
            f"got {type(algorithm.policy).__name__}"
        )
    for layer in layers:
        if len(action_bias) != layer.bias.numel():
            raise ValueError(
                f"initial_action_bias has {len(action_bias)} entries but the policy's action output layer for "
                f"{algorithm.action_space} has {layer.bias.numel()} outputs"
            )
        with th.no_grad():
            layer.bias.copy_(layer.bias.new_tensor(action_bias))
    logging.info("Initialised the policy's action bias to %s", action_bias.tolist())

import sb3_contrib
import stable_baselines3
from torch.optim import Optimizer

SUPPORTED_OPTIMIZER_PATHS: dict[type, tuple[str, ...]] = {
    stable_baselines3.A2C: ("policy.optimizer",),
    stable_baselines3.DDPG: ("policy.actor.optimizer", "policy.critic.optimizer"),
    stable_baselines3.DQN: ("policy.optimizer",),
    stable_baselines3.PPO: ("policy.optimizer",),
    stable_baselines3.SAC: ("policy.actor.optimizer", "policy.critic.optimizer"),
    stable_baselines3.TD3: ("policy.actor.optimizer", "policy.critic.optimizer"),
    sb3_contrib.TRPO: ("policy.optimizer",),
}


def _require_optimizer(root: object, path: str) -> Optimizer:
    current = root
    for part in path.split("."):
        if not hasattr(current, part):
            raise RuntimeError(f"{type(root).__name__} is missing required optimizer attribute '{path}'")
        current = getattr(current, part)

    if not isinstance(current, Optimizer):
        raise RuntimeError(f"{type(root).__name__} has invalid optimizer at '{path}'")

    return current


def get_optimizers_to_reset(algorithm: object) -> dict[str, Optimizer]:
    """Return all optimizer objects whose running state should be cleared."""
    matched_type = next(
        (algo_type for algo_type in SUPPORTED_OPTIMIZER_PATHS if isinstance(algorithm, algo_type)),
        None,
    )
    if matched_type is None:
        raise NotImplementedError(f"Unknown optimizer reset strategy for algorithm class {type(algorithm).__name__}")

    optimizers = {path: _require_optimizer(algorithm, path) for path in SUPPORTED_OPTIMIZER_PATHS[matched_type]}
    if matched_type is stable_baselines3.SAC:
        ent_coef_optimizer = getattr(algorithm, "ent_coef_optimizer", None)
        if ent_coef_optimizer is not None:
            if not isinstance(ent_coef_optimizer, Optimizer):
                raise RuntimeError(f"{type(algorithm).__name__} has invalid optimizer at 'ent_coef_optimizer'")
            optimizers["ent_coef_optimizer"] = ent_coef_optimizer

    return optimizers


def reset_optimizer_state(algorithm: object) -> list[str]:
    """Clear optimizer running averages so training starts with fresh optimizer state."""
    optimizers = get_optimizers_to_reset(algorithm)
    for optimizer in optimizers.values():
        optimizer.state.clear()
    return sorted(optimizers)

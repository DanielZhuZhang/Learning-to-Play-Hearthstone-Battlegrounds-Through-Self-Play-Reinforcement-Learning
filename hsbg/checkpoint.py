"""Save and restore PPO training state."""

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from hsbg.actor import ActorNetwork
from hsbg.critic import CriticNetwork
from hsbg.ppo import PPOConfig


def save_checkpoint(
    path: str | Path,
    actor: ActorNetwork,
    critic: CriticNetwork,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    *,
    iteration: int,
    seed: int,
    ppo_config: PPOConfig,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "actor_optimizer": actor_optimizer.state_dict(),
            "critic_optimizer": critic_optimizer.state_dict(),
            "iteration": iteration,
            "seed": seed,
            "ppo_config": asdict(ppo_config),
        },
        destination,
    )


def load_checkpoint(
    path: str | Path,
    actor: ActorNetwork,
    critic: CriticNetwork,
    *,
    actor_optimizer: torch.optim.Optimizer | None = None,
    critic_optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    state = torch.load(source, map_location=map_location)
    actor.load_state_dict(state["actor"])
    critic.load_state_dict(state["critic"])
    if actor_optimizer is not None:
        actor_optimizer.load_state_dict(state["actor_optimizer"])
    if critic_optimizer is not None:
        critic_optimizer.load_state_dict(state["critic_optimizer"])
    return state

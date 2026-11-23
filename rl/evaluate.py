"""Evaluate a checkpoint deterministically against random opponents."""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hsbg.actor import ActorNetwork  # noqa: E402
from hsbg.checkpoint import load_checkpoint  # noqa: E402
from hsbg.critic import CriticNetwork  # noqa: E402
from hsbg.env import BattlegroundsEnv  # noqa: E402


PROFILE = ROOT / "data" / "profiles" / "initial-15.6.json"


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    episodes: int
    mean_reward: float
    mean_placement: float
    mean_decisions: float


@torch.no_grad()
def evaluate_actor(
    env: BattlegroundsEnv,
    actor: ActorNetwork,
    seeds: Iterable[int],
) -> EvaluationResult:
    actor.eval()
    device = next(actor.parameters()).device
    rewards: list[float] = []
    placements: list[int] = []
    decisions: list[int] = []

    for seed in seeds:
        observation, info = env.reset(seed=seed)
        episode_reward = 0.0
        episode_decisions = 0
        while True:
            obs = torch.as_tensor(observation, dtype=torch.float32, device=device)
            mask = torch.as_tensor(info["action_mask"], dtype=torch.bool, device=device)
            action, _, _ = actor.act(obs, mask, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action.item())
            episode_reward += float(reward)
            episode_decisions += 1
            if terminated or truncated:
                break

        rewards.append(episode_reward)
        placements.append(int(info["placement"]))
        decisions.append(episode_decisions)

    if not rewards:
        raise ValueError("Provide at least one evaluation seed.")
    return EvaluationResult(
        episodes=len(rewards),
        mean_reward=float(np.mean(rewards)),
        mean_placement=float(np.mean(placements)),
        mean_decisions=float(np.mean(decisions)),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a PPO checkpoint.")
    parser.add_argument("checkpoint", nargs="?", type=Path, default=ROOT / "checkpoints" / "ppo_latest.pt")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes < 1:
        raise ValueError("episodes must be positive.")

    env = BattlegroundsEnv.from_profile(PROFILE)
    actor = ActorNetwork.from_env(env)
    critic = CriticNetwork.from_env(env)
    load_checkpoint(args.checkpoint, actor, critic)
    result = evaluate_actor(
        env,
        actor,
        range(args.seed, args.seed + args.episodes),
    )
    print(
        f"episodes={result.episodes} "
        f"mean_reward={result.mean_reward:.4f} "
        f"mean_placement={result.mean_placement:.2f} "
        f"mean_decisions={result.mean_decisions:.1f}"
    )


if __name__ == "__main__":
    main()

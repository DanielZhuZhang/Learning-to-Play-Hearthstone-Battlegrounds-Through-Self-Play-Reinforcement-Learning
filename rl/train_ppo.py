"""PPO training entry point for the Hearthstone Battlegrounds environment."""

import argparse
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hsbg.actor import ActorNetwork  # noqa: E402
from hsbg.critic import CriticNetwork  # noqa: E402
from hsbg.env import BattlegroundsEnv  # noqa: E402
from hsbg.ppo import PPOConfig, update_ppo  # noqa: E402
from hsbg.rollout import collect_rollout  # noqa: E402
from hsbg.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from hsbg.selfplay import SnapshotPool  # noqa: E402


PROFILE = ROOT / "data" / "profiles" / "initial-15.6.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an actor-critic PPO policy.")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--episodes-per-rollout", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument(
        "--self-play",
        action="store_true",
        help="Opponents are frozen snapshots of the learner instead of random bots.",
    )
    parser.add_argument(
        "--pool-size",
        type=int,
        default=1,
        help="Keep this many recent snapshots; 1 = always the previous iteration.",
    )
    parser.add_argument(
        "--random-opponents",
        type=int,
        default=0,
        help="With --self-play, keep this many opponent seats as random bots.",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.iterations < 1 or args.episodes_per_rollout < 1:
        raise ValueError("iterations and episodes-per-rollout must be positive.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available.")

    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    device = torch.device(args.device)

    env = BattlegroundsEnv.from_profile(PROFILE)
    actor = ActorNetwork.from_env(env).to(device)
    critic = CriticNetwork.from_env(env).to(device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
    ppo_config = PPOConfig(
        epochs=4,
        minibatch_size=64,
    )

    next_seed = args.seed
    start_iteration = 0
    if args.resume is not None:
        state = load_checkpoint(
            args.resume,
            actor,
            critic,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            map_location=device,
        )
        start_iteration = int(state["iteration"])
        next_seed = int(state["seed"])
        ppo_config = PPOConfig(**state["ppo_config"])

    checkpoint_dir = args.checkpoint_dir
    if args.save_every < 1:
        raise ValueError("save-every must be positive.")
    if args.pool_size < 1:
        raise ValueError("pool-size must be positive.")
    opponent_seats = env.profile.rules.player_count - 1
    if not 0 <= args.random_opponents <= opponent_seats:
        raise ValueError(f"random-opponents must be in [0, {opponent_seats}].")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    pool = None
    if args.self_play:
        pool = SnapshotPool(args.pool_size, device=device)
        pool.add(actor, iteration=start_iteration)
        opponent_rng = random.Random(next_seed ^ 0x5E1F)
        print(
            f"self-play: pool_size={args.pool_size} "
            f"random_opponents={args.random_opponents}"
        )

    for iteration in range(start_iteration + 1, start_iteration + args.iterations + 1):
        seeds = range(next_seed, next_seed + args.episodes_per_rollout)
        next_seed += args.episodes_per_rollout

        if pool is not None:
            pool.assign(env, opponent_rng, random_seats=args.random_opponents)

        actor.eval()
        critic.eval()
        buffer = collect_rollout(
            env,
            actor,
            critic,
            seeds,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
        )
        actor.train()
        metrics = update_ppo(
            actor,
            critic,
            buffer,
            actor_optimizer,
            critic_optimizer,
            ppo_config,
        )
        if pool is not None:
            pool.add(actor, iteration=iteration)

        mean_return = buffer.rewards.sum().item() / args.episodes_per_rollout
        print(
            f"iteration={iteration} episodes={args.episodes_per_rollout} "
            f"decisions={len(buffer.actions)} mean_episode_return={mean_return:.4f} "
            f"policy_loss={metrics.policy_loss:.6f} "
            f"value_loss={metrics.value_loss:.6f} "
            f"entropy={metrics.entropy:.6f} "
            f"approx_kl={metrics.approximate_kl:.6f}"
        )

        if iteration % args.save_every == 0:
            save_checkpoint(
                checkpoint_dir / f"ppo_{iteration}.pt",
                actor,
                critic,
                actor_optimizer,
                critic_optimizer,
                iteration=iteration,
                seed=next_seed,
                ppo_config=ppo_config,
            )
            save_checkpoint(
                checkpoint_dir / "ppo_latest.pt",
                actor,
                critic,
                actor_optimizer,
                critic_optimizer,
                iteration=iteration,
                seed=next_seed,
                ppo_config=ppo_config,
            )


if __name__ == "__main__":
    main()

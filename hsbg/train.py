"""Train a small PPO policy against fixed bots: python -m hsbg.train --help."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from hsbg.learning import BASELINES, DEFAULT_PROFILE, BattlegroundsEnv, load_content
from hsbg.ppo import (
    ActorCritic,
    PPOConfig,
    collect_episodes,
    save_checkpoint,
    update_policy,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--output", type=Path, default=Path("runs/ppo"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--updates", type=int, default=20)
    parser.add_argument("--episodes-per-update", type=int, default=8)
    parser.add_argument("--opponent", choices=(*BASELINES, "mixed"), default="mixed")
    parser.add_argument("--max-actions", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=128)
    args = parser.parse_args(argv)
    if args.updates < 1 or args.episodes_per_update < 1:
        parser.error("updates and episodes-per-update must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("output directory is not empty; choose a new run directory")
    config = PPOConfig(hidden_size=args.hidden_size)
    random.seed(args.seed)
    np.random.seed(args.seed % (2**32))
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    env = BattlegroundsEnv(
        *load_content(args.profile),
        opponent=args.opponent,
        max_actions=args.max_actions,
    )
    model = ActorCritic(env.encoder.size, env.codec.n_actions, config.hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "training_seed": args.seed,
        "opponent": args.opponent,
        "episodes": 0,
        "decisions": 0,
        "torch_version": str(torch.__version__),
        "numpy_version": str(np.__version__),
    }
    for update in range(1, args.updates + 1):
        batch, results = collect_episodes(
            env,
            model,
            args.episodes_per_update,
            args.seed + metadata["episodes"],
            config,
            seat_offset=metadata["episodes"],
        )
        losses = update_policy(model, optimizer, batch, config)
        metadata["episodes"] += args.episodes_per_update
        metadata["decisions"] += len(batch["actions"])
        record = {
            "update": update,
            **metadata,
            **losses,
            "train_mean_placement": float(np.mean([r["placement"] for r in results])),
            "train_top_four_credit": float(
                np.mean([r["top_four_credit"] for r in results])
            ),
        }
        with (args.output / "training.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        save_checkpoint(args.output / "policy.pt", model, config, env, metadata)
        print(json.dumps(record), flush=True)
    print(
        "Training statistics are not held-out performance. Run python -m hsbg.evaluate."
    )


if __name__ == "__main__":
    main()

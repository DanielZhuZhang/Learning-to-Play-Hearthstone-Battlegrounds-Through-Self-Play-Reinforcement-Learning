"""Frozen-policy evaluation on a specified seed range and rotating learner seats."""

import argparse
import json
import random
from pathlib import Path

import numpy as np

from hsbg.learning import (
    BASELINES,
    DEFAULT_PROFILE,
    BattlegroundsEnv,
    baseline_action,
    load_content,
)


def evaluate(env, policy, games, seed, *, model=None):
    if games < 1:
        raise ValueError("games must be positive")
    if policy not in BASELINES and not (policy == "ppo" and model is not None):
        raise ValueError("Specify a baseline or provide a PPO model")
    records = []
    for game in range(games):
        game_seed = seed + game
        seat = game % env.profile.rules.player_count
        rng = random.Random(game_seed ^ 0xE7A1)
        obs, _ = env.reset(seed=game_seed, seat=seat)
        done, decisions = False, 0
        while not done:
            action = (
                model.act(obs, deterministic=True)[0]
                if model is not None
                else baseline_action(obs, env.encoder, rng, policy)
            )
            obs, _, done, _, info = env.step(action)
            decisions += 1
        records.append(
            {"seed": game_seed, "seat": seat, "decisions": decisions, **info}
        )
    placements = [r["placement"] for r in records]
    return {
        "policy": policy,
        "opponent": env.opponent,
        "games": games,
        "mean_placement": float(np.mean(placements)),
        "placement_std": float(np.std(placements, ddof=1)) if games > 1 else None,
        "top_four_rate": float(np.mean([r["top_four_credit"] for r in records])),
        "match_turn_limit_rate": float(
            np.mean([r["match_turn_limit"] for r in records])
        ),
        "contract": env.contract(),
        "episodes": records,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--policy", choices=(*BASELINES, "ppo"), default="greedy")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--opponent", choices=(*BASELINES, "mixed"), default="mixed")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=100000)
    parser.add_argument("--max-actions", type=int, default=64)
    parser.add_argument("--output", type=Path, default=Path("runs/evaluation.json"))
    args = parser.parse_args(argv)
    if args.games < 1:
        parser.error("games must be positive")
    if (args.policy == "ppo") != (args.checkpoint is not None):
        parser.error("--checkpoint is required exactly when --policy ppo")
    if args.output.exists():
        parser.error("output file already exists; choose a new path")
    env = BattlegroundsEnv(
        *load_content(args.profile),
        opponent=args.opponent,
        max_actions=args.max_actions,
    )
    model, metadata = None, None
    if args.policy == "ppo":
        import torch

        from hsbg.ppo import load_checkpoint

        torch.set_num_threads(1)
        model, metadata = load_checkpoint(args.checkpoint, env)
        train_start = metadata["training_seed"]
        train_end = train_start + metadata["episodes"]
        if args.seed < train_end and args.seed + args.games > train_start:
            parser.error("evaluation seeds overlap the checkpoint training seed range")
    report = evaluate(env, args.policy, args.games, args.seed, model=model)
    report["checkpoint_metadata"] = metadata
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "episodes"}, indent=2))


if __name__ == "__main__":
    main()

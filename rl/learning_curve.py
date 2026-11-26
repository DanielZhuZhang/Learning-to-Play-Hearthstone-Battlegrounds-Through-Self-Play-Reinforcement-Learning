"""Score every saved checkpoint of a run to see how fast training improves.

For each ``ppo_<iteration>.pt`` in a run directory this reports:

* placement against 7 random bots (random play averages 4.5);
* placement against 7 copies of a fixed reference checkpoint (default: the
  highest-numbered ``ppo_<N>.pt`` present when the script starts, so a run
  that is still training cannot change it mid-evaluation), which keeps
  showing progress after the agent beats random bots every game;
* how many decisions per game are padding (moves and freeze toggles).

By default the learner samples its actions, as in training and as the
reference opponents do, so a checkpoint against copies of itself averages
4.5. ``--greedy`` makes the learner always take its top choice instead; this
policy was trained by sampling and plays noticeably worse greedily. Every
checkpoint plays the same seeds. Checkpoints are scored in parallel.

    python rl/learning_curve.py checkpoints/cluster_selfplay_s0 --episodes 50 --workers 16
    python rl/learning_curve.py checkpoints/cluster_selfplay_s0 --every 4 --csv curve_s0.csv
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hsbg.actor import ActorNetwork  # noqa: E402
from hsbg.critic import CriticNetwork  # noqa: E402
from hsbg.env import BattlegroundsEnv  # noqa: E402
from hsbg.selfplay import ActorPolicy  # noqa: E402

PROFILE = ROOT / "data" / "profiles" / "initial-15.6.json"
PADDING = ("move", "freeze")


def _load_actor(env: BattlegroundsEnv, path: Path) -> ActorNetwork:
    actor = ActorNetwork.from_env(env)
    critic = CriticNetwork.from_env(env)
    state = torch.load(path, map_location="cpu")
    actor.load_state_dict(state["actor"])
    critic.load_state_dict(state["critic"])
    return actor.eval()


@torch.no_grad()
def _play(
    env: BattlegroundsEnv, actor: ActorNetwork, seeds: range, greedy: bool
) -> tuple[float, float, float]:
    """Mean placement, decisions per game, and padding share of decisions."""
    placements, decisions, padding = [], 0, 0
    for seed in seeds:
        obs, info = env.reset(seed=seed)
        while True:
            action, _, _ = actor.act(
                torch.as_tensor(obs, dtype=torch.float32),
                torch.as_tensor(info["action_mask"], dtype=torch.bool),
                deterministic=greedy,
            )
            index = int(action.item())
            decisions += 1
            if env.describe_action(index).startswith(PADDING):
                padding += 1
            obs, _, terminated, truncated, info = env.step(index)
            if terminated or truncated:
                break
        placements.append(info["placement"])
    return float(np.mean(placements)), decisions / len(seeds), padding / max(decisions, 1)


def _score(job: tuple[str, str | None, int, int, bool]) -> dict[str, float | int | str]:
    checkpoint, reference, episodes, seed, greedy = job
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    env = BattlegroundsEnv.from_profile(PROFILE)
    actor = _load_actor(env, Path(checkpoint))
    seeds = range(seed, seed + episodes)

    row: dict[str, float | int | str] = {"checkpoint": Path(checkpoint).name}
    row["vs_random"], row["decisions_per_game"], row["padding_share"] = _play(env, actor, seeds, greedy)
    if reference is not None:
        opponent = ActorPolicy(_load_actor(env, Path(reference)))
        env.opponents = {
            f"player-{i + 1}": opponent
            for i in range(env.profile.rules.player_count)
            if f"player-{i + 1}" != env.learner_id
        }
        torch.manual_seed(seed + 1)
        row["vs_reference"], _, _ = _play(env, actor, seeds, greedy)
    return row


def _iteration(path: Path) -> int:
    match = re.fullmatch(r"ppo_(\d+)\.pt", path.name)
    return int(match.group(1)) if match else -1


def main() -> None:
    parser = argparse.ArgumentParser(description="Score every checkpoint of a training run.")
    parser.add_argument("run_dir", type=Path, help="e.g. checkpoints/cluster_selfplay_s0")
    parser.add_argument("--episodes", type=int, default=50, help="games per checkpoint per opponent")
    parser.add_argument("--seed", type=int, default=5000)
    parser.add_argument("--every", type=int, default=1, help="score every Nth checkpoint (latest always included)")
    parser.add_argument("--reference", type=Path, default=None,
                        help="opponent checkpoint (default: highest-numbered ppo_<N>.pt in run_dir)")
    parser.add_argument("--no-reference", action="store_true", help="only score against random bots")
    parser.add_argument("--greedy", action="store_true",
                        help="learner always takes its top choice instead of sampling")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--csv", type=Path, default=None, help="also write the table to this CSV file")
    args = parser.parse_args()

    checkpoints = sorted((p for p in args.run_dir.glob("ppo_*.pt") if _iteration(p) >= 0), key=_iteration)
    if not checkpoints:
        parser.error(f"No ppo_<iteration>.pt files in {args.run_dir}")
    selected = checkpoints[:: max(1, args.every)]
    if selected[-1] != checkpoints[-1]:
        selected.append(checkpoints[-1])

    reference = None
    if not args.no_reference:
        reference = args.reference or checkpoints[-1]
        if not reference.is_file():
            parser.error(f"Reference checkpoint not found: {reference}")
        print(f"reference opponent: {reference} (7 copies, sampling)")
    print(f"learner: {'top choice (greedy)' if args.greedy else 'sampling, as in training'}")
    print(f"scoring {len(selected)} checkpoints x {args.episodes} games with {args.workers} workers\n")

    jobs = [(str(p), str(reference) if reference else None, args.episodes, args.seed, args.greedy)
            for p in selected]
    header = f"{'iteration':>9}  {'vs_random':>9}  {'vs_reference':>12}  {'decisions/game':>14}  {'padding':>7}"
    print(header)
    print("-" * len(header))
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as pool:
        for path, row in zip(selected, pool.map(_score, jobs)):
            row["iteration"] = _iteration(path)
            rows.append(row)
            ref = f"{row['vs_reference']:12.2f}" if "vs_reference" in row else f"{'-':>12}"
            print(f"{row['iteration']:>9}  {row['vs_random']:9.2f}  {ref}  "
                  f"{row['decisions_per_game']:14.0f}  {row['padding_share']:7.0%}", flush=True)

    print("\nPlacement: 1 = always first, 8 = always last. Random play and an equal opponent both average 4.5.")
    print(f"With {args.episodes} games, differences under about {4.6 / args.episodes ** 0.5:.1f} places are within noise.")
    if args.csv:
        with args.csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["iteration", "checkpoint", "vs_random", "vs_reference",
                                                        "decisions_per_game", "padding_share"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()

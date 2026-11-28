"""Round-robin tournament between checkpoints.

Every checkpoint plays ``--episodes`` games against 7 copies of every
checkpoint (itself included). Everyone samples actions, as in training.
Cell (row i, column j) is i's mean placement against 7 copies of j:
4.5 is an even match, lower is better. The diagonal (a checkpoint against
itself) should be near 4.5 and checks that the setup is fair.

Each checkpoint is ranked by its mean placement over the other columns.

    python rl/tournament.py ppo_5000_s0.pt ppo_5000_s1.pt ppo_5000_s2.pt --episodes 200 --workers 12
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hsbg.actor import ActorNetwork  # noqa: E402
from hsbg.env import BattlegroundsEnv  # noqa: E402
from hsbg.selfplay import ActorPolicy  # noqa: E402

PROFILE = ROOT / "data" / "profiles" / "initial-15.6.json"


def _load(env: BattlegroundsEnv, path: str) -> ActorNetwork:
    actor = ActorNetwork.from_env(env)
    actor.load_state_dict(torch.load(path, map_location="cpu")["actor"])
    return actor.eval()


@torch.no_grad()
def _match(job: tuple[int, int, str, str, int, int]) -> tuple[int, int, list[int]]:
    """Learner ``row`` against 7 copies of ``col``; returns its placements."""
    row, col, learner_path, opponent_path, episodes, seed = job
    torch.set_num_threads(1)
    torch.manual_seed(seed * 7 + 1000 * row + col)
    env = BattlegroundsEnv.from_profile(PROFILE)
    learner = _load(env, learner_path)
    opponent = ActorPolicy(_load(env, opponent_path))
    env.opponents = {
        f"player-{i + 1}": opponent
        for i in range(env.profile.rules.player_count)
        if f"player-{i + 1}" != env.learner_id
    }
    placements = []
    for game_seed in range(seed, seed + episodes):
        obs, info = env.reset(seed=game_seed)
        while True:
            action, _, _ = learner.act(
                torch.as_tensor(obs, dtype=torch.float32),
                torch.as_tensor(info["action_mask"], dtype=torch.bool),
            )
            obs, _, terminated, truncated, info = env.step(int(action.item()))
            if terminated or truncated:
                break
        placements.append(int(info["placement"]))
    return row, col, placements


def main() -> None:
    p = argparse.ArgumentParser(description="Round-robin tournament between checkpoints.")
    p.add_argument("checkpoints", type=Path, nargs="+")
    p.add_argument("--episodes", type=int, default=200, help="games per pairing")
    p.add_argument("--seed", type=int, default=7000)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--csv", type=Path, help="also write the matrix to this CSV file")
    p.add_argument("--names", nargs="+", help="short label for each checkpoint, in the same order")
    args = p.parse_args()

    paths = [str(c) for c in args.checkpoints]
    n = len(paths)
    if args.names:
        if len(args.names) != n:
            p.error(f"--names has {len(args.names)} labels for {n} checkpoints")
        names = list(args.names)
    else:
        names = [c.stem for c in args.checkpoints]
        if len(set(names)) < n:  # e.g. several ppo_7000.pt: label by run directory
            names = [f"{c.parent.name}/{c.stem}" for c in args.checkpoints]
    for path in args.checkpoints:
        if not path.is_file():
            p.error(f"Checkpoint not found: {path}")

    # Split each pairing's games into chunks so all workers stay busy.
    chunk = max(1, min(25, args.episodes))
    jobs = [(i, j, paths[i], paths[j], min(chunk, args.episodes - start), args.seed + start)
            for i in range(n) for j in range(n) for start in range(0, args.episodes, chunk)]
    print(f"{n} checkpoints, {n * n} pairings x {args.episodes} games, {args.workers} workers\n", flush=True)
    results: dict[tuple[int, int], list[int]] = {}
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as pool:
        for row, col, placements in pool.map(_match, jobs):
            results.setdefault((row, col), []).extend(placements)
    matrix = np.array([[np.mean(results[i, j]) for j in range(n)] for i in range(n)])
    firsts = np.array([[np.mean(np.array(results[i, j]) == 1) for j in range(n)] for i in range(n)])

    labels = names if args.names else [name.rsplit("_", 1)[-1] for name in names]
    if len(set(labels)) < n:
        labels = [f"c{i}" for i in range(n)]
    col = max(6, *(len(label) for label in labels))
    width = max(len(name) for name in names)
    print("Mean placement of the row checkpoint against 7 copies of the column checkpoint")
    print("(4.5 = even, lower = better; diagonal = self-play check)\n")
    if labels != names and not args.names:
        print("columns: " + ", ".join(f"{label} = {name}" for label, name in zip(labels, names)) + "\n")
    print(" " * width + "  " + "  ".join(f"{label:>{col}}" for label in labels) + "  vs_others")
    off_diagonal = ~np.eye(n, dtype=bool)
    scores = [matrix[i][off_diagonal[i]].mean() if n > 1 else matrix[i, i] for i in range(n)]
    for i, name in enumerate(names):
        cells = "  ".join(f"{matrix[i, j]:{col}.2f}" for j in range(n))
        print(f"{name:>{width}}  {cells}  {scores[i]:9.2f}")

    print(f"\nWith {args.episodes} games per cell, differences under about "
          f"{4.6 / args.episodes ** 0.5:.2f} places are noise; the vs_others average is "
          f"about {4.6 / (args.episodes * max(n - 1, 1)) ** 0.5:.2f}.")
    print(f"Diagonal (should be near 4.5): {', '.join(f'{matrix[i, i]:.2f}' for i in range(n))}\n")
    print("Ranking (lower = stronger):")
    for rank, i in enumerate(np.argsort(scores), start=1):
        print(f"  {rank}. {names[i]}  {scores[i]:.2f}  (first place in {firsts[i][off_diagonal[i]].mean():.0%} of games vs others)")

    if args.csv:
        with args.csv.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["checkpoint", *names, "vs_others"])
            for i, name in enumerate(names):
                writer.writerow([name, *(f"{v:.3f}" for v in matrix[i]), f"{scores[i]:.3f}"])
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()

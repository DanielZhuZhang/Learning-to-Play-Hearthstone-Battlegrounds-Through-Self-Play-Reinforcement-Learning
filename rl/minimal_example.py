"""Minimal example of interacting with the Battlegrounds RL environment.

Plays one episode as ``player-1`` by picking a random *legal* action each step
(7 random bots play the other seats). Replace ``choose_action`` with your
policy.

Run from the repository root:

    python rl/minimal_example.py
"""

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # lets the script import hsbg without installing it

from hsbg.env import BattlegroundsEnv  # noqa: E402
from hsbg.actor import ActorNetwork  # noqa: E402

def choose_action(
    observation: np.ndarray,
    action_mask: np.ndarray,
    actor: ActorNetwork,
) -> int:
    observation_tensor = torch.from_numpy(observation).float()
    mask_tensor = torch.from_numpy(action_mask).bool()

    with torch.no_grad():
        action, _, _ = actor.act(
            observation_tensor,
            mask_tensor,
            deterministic=False,
        )

    return int(action.item())

def main() -> None:
    env = BattlegroundsEnv.from_profile(ROOT / "data" / "profiles" / "initial-15.6.json")
    print(f"observation size: {env.observation_size}, action size: {env.action_size}")

    actor = ActorNetwork.from_env(env).eval()
    observation, info = env.reset(seed=0)
    total_reward = 0.0
    steps = 0
    terminated = False

    while not terminated:
        action = choose_action(observation, info["action_mask"], actor)
        if info["turn"] <= 2:  # print the first couple of turns only
            print(f"turn {info['turn']}: {env.describe_action(action)}")

        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

    me = env.engine.player_state("player-1")
    print(
        f"finished in place {info['placement']} after {steps} steps "
        f"and {info['turn']} turns (health {me.health}); reward {total_reward:+.3f}"
    )


if __name__ == "__main__":
    main()

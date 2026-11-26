from pathlib import Path

import numpy as np
import pytest
import torch

from hsbg.actor import ActorNetwork
from hsbg.critic import CriticNetwork
from hsbg.env import BattlegroundsEnv
from hsbg.game.legal import RecruitActionKind
from hsbg.parallel import ParallelRolloutCollector

PROFILE = Path(__file__).resolve().parents[1] / "data" / "profiles" / "initial-15.6.json"


def play_scripted(env, seed=3, steps=200):
    """Take the first legal non-end action each step (end turn when none); return (kind, reward) pairs."""
    obs, info = env.reset(seed=seed)
    out = []
    for _ in range(steps):
        legal = np.flatnonzero(info["action_mask"])
        choice = next((i for i in legal if i != env.action_layout.end_turn), env.action_layout.end_turn)
        kind = env._moves[int(choice)].action.kind
        obs, reward, done, _, info = env.step(int(choice))
        out.append((kind, reward, done))
        if done:
            break
    return out


def test_default_has_no_penalty():
    steps = play_scripted(BattlegroundsEnv.from_profile(PROFILE))
    assert all(reward == 0.0 for _, reward, done in steps if not done)


@pytest.mark.parametrize("mode", ["all", "padding"])
def test_penalty_charges_only_selected_actions(mode):
    env = BattlegroundsEnv.from_profile(PROFILE, action_penalty=0.01, penalized_actions=mode)
    steps = play_scripted(env)
    padding = {RecruitActionKind.REORDER_BOARD, RecruitActionKind.FREEZE, RecruitActionKind.UNFREEZE}
    charged = 0
    for kind, reward, done in steps:
        if done:
            continue
        if kind is RecruitActionKind.END_RECRUIT or (mode == "padding" and kind not in padding):
            assert reward == 0.0
        else:
            assert reward == pytest.approx(-0.01)
            charged += 1
    assert charged > 0


def test_invalid_penalty_settings_are_rejected():
    with pytest.raises(ValueError):
        BattlegroundsEnv.from_profile(PROFILE, action_penalty=-1)
    with pytest.raises(ValueError):
        BattlegroundsEnv.from_profile(PROFILE, penalized_actions="moves")


def test_parallel_workers_apply_the_penalty():
    env = BattlegroundsEnv.from_profile(PROFILE)
    torch.manual_seed(0)
    actor, critic = ActorNetwork.from_env(env).eval(), CriticNetwork.from_env(env).eval()
    with ParallelRolloutCollector(PROFILE, workers=1, env_kwargs={"action_penalty": 0.01}) as pool:
        buffer = pool.collect(actor, critic, [60])
    non_terminal = buffer.rewards[~buffer.terminated]
    charged = torch.isclose(non_terminal, torch.tensor(-0.01))
    assert bool(((non_terminal == 0) | charged).all())
    assert bool(charged.any())

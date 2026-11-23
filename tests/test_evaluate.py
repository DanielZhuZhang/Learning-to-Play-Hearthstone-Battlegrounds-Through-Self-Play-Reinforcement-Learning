from pathlib import Path

from hsbg.actor import ActorNetwork
from hsbg.env import BattlegroundsEnv
from rl.evaluate import evaluate_actor


PROFILE = Path(__file__).resolve().parents[1] / "data" / "profiles" / "initial-15.6.json"


def test_deterministic_evaluation_runs_complete_episodes():
    env = BattlegroundsEnv.from_profile(PROFILE)
    actor = ActorNetwork.from_env(env).eval()

    result = evaluate_actor(env, actor, [0, 1])

    assert result.episodes == 2
    assert 1 <= result.mean_placement <= env.profile.rules.player_count
    assert result.mean_decisions > 0

from pathlib import Path
import random

import pytest

torch = pytest.importorskip("torch")

from hsbg.actor import ActorConfig, ActorNetwork  # noqa: E402
from hsbg.data.profiles import load_game_data  # noqa: E402
from hsbg.env import BattlegroundsEnv  # noqa: E402
from hsbg.selfplay import ActorPolicy, SnapshotPool  # noqa: E402

PROFILE = Path(__file__).resolve().parents[1] / "data" / "profiles" / "initial-15.6.json"
SMALL = ActorConfig(hidden_sizes=(32,))


@pytest.fixture(scope="module")
def content():
    return load_game_data(PROFILE)


@pytest.fixture()
def env(content):
    return BattlegroundsEnv(*content)


@pytest.fixture()
def actor(env):
    torch.manual_seed(0)
    return ActorNetwork.from_env(env, SMALL)


def test_pool_keeps_only_most_recent_snapshots(actor):
    pool = SnapshotPool(2)
    for iteration in range(5):
        pool.add(actor, iteration)
    assert pool.iterations == [3, 4]


def test_snapshot_is_frozen_against_later_training(actor):
    pool = SnapshotPool(1)
    pool.add(actor, 0)
    snapshot = pool.snapshots[0][1].actor
    before = [p.clone() for p in snapshot.parameters()]
    with torch.no_grad():
        for p in actor.parameters():
            p.add_(1.0)
    assert all(torch.equal(a, b) for a, b in zip(before, snapshot.parameters()))


def test_assign_fills_every_opponent_seat(env, actor):
    pool = SnapshotPool(3)
    pool.add(actor, 7)
    chosen = pool.assign(env, random.Random(0))
    seats = env.profile.rules.player_count - 1
    assert len(chosen) == seats
    assert env.learner_id not in chosen
    assert all(isinstance(env.opponents[seat], ActorPolicy) for seat in chosen)
    assert set(chosen.values()) == {7}


def test_assign_keeps_requested_random_seats(env, actor):
    pool = SnapshotPool(1)
    pool.add(actor, 0)
    pool.assign(env, random.Random(0))
    chosen = pool.assign(env, random.Random(1), random_seats=2)
    random_seats = [seat for seat, it in chosen.items() if it is None]
    assert len(random_seats) == 2
    assert all(seat not in env.opponents for seat in random_seats)


def test_self_play_episode_runs_to_completion(env, actor):
    pool = SnapshotPool(1)
    pool.add(actor, 0)
    pool.assign(env, random.Random(0))
    obs, info = env.reset(seed=3)
    rng = random.Random(3)
    for _ in range(5000):
        legal = [i for i, ok in enumerate(info["action_mask"]) if ok]
        obs, reward, terminated, truncated, info = env.step(rng.choice(legal))
        if terminated or truncated:
            break
    assert terminated
    assert info["placement"] is not None

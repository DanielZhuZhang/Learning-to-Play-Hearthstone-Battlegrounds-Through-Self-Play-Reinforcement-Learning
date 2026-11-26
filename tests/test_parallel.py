import random
from pathlib import Path

import pytest
import torch

from hsbg.actor import ActorNetwork
from hsbg.critic import CriticNetwork
from hsbg.env import BattlegroundsEnv
from hsbg.parallel import ParallelRolloutCollector, split_seeds
from hsbg.rollout import collect_rollout
from hsbg.selfplay import SnapshotPool

PROFILE = Path(__file__).resolve().parents[1] / "data" / "profiles" / "initial-15.6.json"


@pytest.fixture(scope="module")
def env():
    return BattlegroundsEnv.from_profile(PROFILE)


@pytest.fixture(scope="module")
def nets(env):
    torch.manual_seed(0)
    return ActorNetwork.from_env(env).eval(), CriticNetwork.from_env(env).eval()


@pytest.fixture(scope="module")
def collector():
    with ParallelRolloutCollector(PROFILE, workers=2) as pool:
        yield pool


def test_split_seeds_is_ordered_and_balanced():
    assert split_seeds(range(7), 3) == [[0, 1, 2], [3, 4], [5, 6]]
    assert split_seeds([4, 5], 8) == [[4], [5]]


def test_single_chunk_matches_serial_collection(env, nets):
    actor, critic = nets
    torch.manual_seed(30)
    serial = collect_rollout(env, actor, critic, [30])
    with ParallelRolloutCollector(PROFILE, workers=1) as pool:
        parallel = pool.collect(actor, critic, [30])
    assert torch.equal(serial.actions, parallel.actions)
    assert torch.equal(serial.observations, parallel.observations)
    assert torch.allclose(serial.advantages, parallel.advantages)


def test_parallel_rollout_keeps_episodes_in_seed_order(env, nets, collector):
    actor, critic = nets
    buffer = collector.collect(actor, critic, [40, 41, 42])
    assert int(buffer.terminated.sum()) == 3
    assert bool(buffer.terminated[-1])
    rows = len(buffer.actions)
    for name in ("observations", "action_masks", "old_log_probs", "advantages", "value_targets"):
        assert len(getattr(buffer, name)) == rows
    # first episode came from seed 40: replaying it serially gives the same first observation
    first_obs, _ = env.reset(seed=40)
    assert torch.allclose(buffer.observations[0], torch.as_tensor(first_obs, dtype=torch.float32))


def test_parallel_self_play_uses_snapshot_opponents(env, nets, collector):
    actor, critic = nets
    pool = SnapshotPool(2)
    pool.add(actor, iteration=0)
    pool.add(actor, iteration=1)
    assignment = pool.draw(env, random.Random(0), random_seats=3)
    assert sum(index is None for index in assignment.values()) == 3
    buffer = collector.collect(actor, critic, [50, 51], pool=pool, assignment=assignment)
    assert int(buffer.terminated.sum()) == 2


def test_draw_matches_assign(env, nets):
    actor, _ = nets
    pool = SnapshotPool(3)
    for iteration in range(3):
        pool.add(actor, iteration=iteration)
    drawn = pool.draw(env, random.Random(5), random_seats=2)
    assigned = pool.assign(env, random.Random(5), random_seats=2)
    assert {seat: (None if i is None else pool.snapshots[i][0]) for seat, i in drawn.items()} == assigned
    env.opponents.clear()

import random
from dataclasses import replace

import numpy as np
import pytest

from hsbg.evaluate import evaluate
from hsbg.game.engine import GameEngine
from hsbg.game.legal import RecruitActionKind
from hsbg.learning import (
    BattlegroundsEnv,
    baseline_action,
    load_content,
    placement_metrics,
)


@pytest.fixture(scope="module")
def content():
    return load_content()


def test_codec_covers_engine_actions_except_reorder(content):
    env = BattlegroundsEnv(*content)
    obs, _ = env.reset(seed=7)
    assert env.codec.n_actions == 29
    for _ in range(80):
        options = env.codec.available(env.engine, env.player_id)
        legal = env.engine.legal_actions(env.player_id)
        assert set(options.values()) == {
            a for a in legal if a.kind is not RecruitActionKind.REORDER_BOARD
        }
        assert obs["features"].shape == (env.encoder.size,)
        assert np.isfinite(obs["features"]).all()
        action = baseline_action(obs, env.encoder, random.Random(3), "greedy")
        obs, _, done, _, _ = env.step(action)
        if done:
            break


@pytest.mark.parametrize("strategy", ["random", "greedy", "upgrade"])
def test_all_baselines_finish_rotating_seats_and_are_reproducible(content, strategy):
    profile, cards = content
    profile = replace(profile, rules=replace(profile.rules, max_match_turns=8))
    env = BattlegroundsEnv(profile, cards, max_actions=8)
    first = evaluate(env, strategy, games=8, seed=100)
    second = evaluate(env, strategy, games=8, seed=100)
    assert first == second
    assert {r["seat"] for r in first["episodes"]} == set(range(8))
    assert all(1 <= r["placement"] <= 8 for r in first["episodes"])
    assert 0 <= first["top_four_rate"] <= 1


def test_observation_does_not_leak_opponent_zones(content):
    env = BattlegroundsEnv(*content)
    before, _ = env.reset(seed=9)
    players = list(env.engine.state.players)
    players[1] = replace(players[1], shop=(), minions=(), gold=0, shop_frozen=True)
    env.engine.state = replace(env.engine.state, players=tuple(players))
    after = env._observe(env.player_id)
    np.testing.assert_array_equal(before["features"], after["features"])
    np.testing.assert_array_equal(before["action_mask"], after["action_mask"])


@pytest.mark.parametrize("action", [-1, 1000, True, 0.0])
def test_bad_actions_do_not_mutate_engine(content, action):
    env = BattlegroundsEnv(*content)
    env.reset(seed=1)
    state, history = env.engine.state, list(env.engine.action_history)
    with pytest.raises(ValueError):
        env.step(action)
    assert env.engine.state == state
    assert env.engine.action_history == history


def test_action_cap_masks_free_action_loops(content):
    env = BattlegroundsEnv(*content, max_actions=2)
    env.reset(seed=1)
    env.step(2)  # freeze
    obs, *_ = env.step(3)  # unfreeze
    assert np.flatnonzero(obs["action_mask"]).tolist() == [0]
    with pytest.raises(ValueError):
        env.step(2)
    obs, *_ = env.step(0)
    assert env.engine.state.turn == 2
    assert obs["action_mask"][2]


def test_terminal_midrank_reward_replay_and_reset(content):
    profile, cards = content
    profile = replace(profile, rules=replace(profile.rules, max_match_turns=1))
    env = BattlegroundsEnv(profile, cards, max_actions=1)
    env.reset(seed=2)
    obs, reward, done, truncated, info = env.step(0)
    assert done and not truncated
    assert not obs["action_mask"].any()
    assert -1 <= reward <= 1
    assert info["match_turn_limit"]
    with pytest.raises(RuntimeError):
        env.step(0)
    replay = GameEngine.replay(profile, cards, env.engine.to_replay())
    assert replay.state == env.engine.state
    assert replay.placements == env.engine.placements
    obs, _ = env.reset(seed=2)
    assert obs["action_mask"].any()


def test_tie_across_fourth_place_splits_credit():
    rank, credit, reward = placement_metrics(
        {"a": 1, "b": 2, "c": 3, "d": 4, "e": 4, "f": 6, "g": 7, "h": 8}, "d", 8
    )
    assert rank == 4.5 and credit == 0.5 and reward == 0
    all_tied = {str(i): 1 for i in range(8)}
    assert placement_metrics(all_tied, "0", 8) == (4.5, 0.5, 0)


def test_episode_ends_at_learner_elimination(content):
    profile, cards = content
    profile = replace(profile, rules=replace(profile.rules, starting_health=1))
    env = BattlegroundsEnv(profile, cards, opponent="greedy")
    env.reset(seed=7)
    for _ in range(100):
        obs, reward, done, _, info = env.step(0)
        if done:
            assert env.player_id in dict(env.engine.placements)
            assert info["placement"] >= 1
            assert not obs["action_mask"].any()
            assert -1 <= reward <= 1
            break
    else:
        pytest.fail("Eliminated learner did not terminate")

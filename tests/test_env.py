from pathlib import Path
import random

import numpy as np
import pytest

from hsbg.data.profiles import load_game_data
from hsbg.env import ActionLayout, BattlegroundsEnv, placement_reward
from hsbg.game.legal import RecruitActionKind

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROFILE = PROJECT_ROOT / "data" / "profiles" / "initial-15.6.json"


@pytest.fixture(scope="module")
def content():
    return load_game_data(PROFILE)


def _random_index(mask: np.ndarray, rng: random.Random) -> int:
    return rng.choice(np.flatnonzero(mask).tolist())


def test_default_layout_has_112_indices_and_readable_names(content) -> None:
    profile, _ = content
    layout = ActionLayout.from_rules(profile.rules)

    assert layout.size == 112
    assert layout.describe(0) == "buy shop slot 0"
    assert layout.describe(layout.play_targeted + 7 + 2) == (
        "play hand slot 1 targeting board slot 2"
    )
    assert layout.describe(layout.end_turn) == "end turn"
    with pytest.raises(ValueError):
        layout.describe(layout.size)


def test_reset_returns_observation_and_mask(content) -> None:
    env = BattlegroundsEnv(*content)
    obs, info = env.reset(seed=3)

    assert obs.shape == (env.observation_size,) and obs.dtype == np.float32
    mask = info["action_mask"]
    assert mask.shape == (env.action_size,) and mask.dtype == bool
    assert mask[env.action_layout.end_turn]
    assert info["turn"] == 1 and info["placement"] is None
    sections = list(env.encoder.sections.values())
    assert sections[0].start == 0 and sections[-1].stop == env.observation_size
    assert all(a.stop == b.start for a, b in zip(sections, sections[1:]))
    shop = obs[env.encoder.sections["shop"]].reshape(-1, env.encoder.card_features)
    player = env.engine.player_state("player-1")
    assert shop[: len(player.shop), 0].tolist() == [1.0] * len(player.shop)
    assert not shop[len(player.shop) :].any()


def test_mask_matches_engine_legal_actions_through_episodes(content) -> None:
    env = BattlegroundsEnv(*content)
    for seed in range(4):
        rng = random.Random(seed)
        env.reset(seed=seed)
        terminated = False
        while not terminated:
            engine = env.engine
            legal = engine.legal_actions("player-1")
            _, mask, moves = env.observe("player-1")
            if env.action_mask().sum() > 1:  # not forced to end by the action cap
                assert np.array_equal(mask, env.action_mask())
            mapped = {m.action for m in moves.values()}
            assert mapped == set(legal)
            for index, move in moves.items():
                if move.action.kind is RecruitActionKind.REORDER_BOARD:
                    board = [m.instance_id for m in engine.player_state("player-1").board]
                    assert sorted(move.board_order) == sorted(board)
                    assert move.board_order != tuple(board)
            _, reward, terminated, truncated, info = env.step(
                _random_index(env.action_mask(), rng)
            )
            assert not truncated
        assert info["placement"] is not None
        assert reward == placement_reward(info["placement"], 8)
        assert not info["action_mask"].any()


def test_episodes_are_deterministic_for_seed_and_actions(content) -> None:
    def run(seed: int) -> list[tuple[bytes, float]]:
        env = BattlegroundsEnv(*content)
        rng = random.Random(99)
        obs, info = env.reset(seed=seed)
        trace = [(obs.tobytes(), 0.0)]
        done = False
        while not done:
            obs, reward, done, _, info = env.step(_random_index(info["action_mask"], rng))
            trace.append((obs.tobytes(), reward))
        return trace

    assert run(5) == run(5)


def test_illegal_and_post_episode_steps_are_rejected(content) -> None:
    env = BattlegroundsEnv(*content)
    with pytest.raises(RuntimeError, match="reset"):
        env.step(0)
    _, info = env.reset(seed=1)
    illegal = int(np.flatnonzero(~info["action_mask"])[0])
    with pytest.raises(ValueError, match="not legal"):
        env.step(illegal)

    done = False
    while not done:
        _, _, done, _, _ = env.step(env.action_layout.end_turn)
    with pytest.raises(RuntimeError, match="episode is over"):
        env.step(env.action_layout.end_turn)


def test_learner_is_limited_to_end_turn_after_action_cap(content) -> None:
    env = BattlegroundsEnv(*content, max_actions_per_turn=2)
    _, info = env.reset(seed=2)
    freeze = env.action_layout.freeze
    _, _, _, _, info = env.step(freeze)
    assert info["action_mask"].sum() > 1
    _, _, _, _, info = env.step(freeze)
    assert np.flatnonzero(info["action_mask"]).tolist() == [env.action_layout.end_turn]
    _, _, _, _, info = env.step(env.action_layout.end_turn)
    assert info["turn"] == 2 and info["action_mask"].sum() > 1


def test_index_policy_opponents_receive_their_own_view(content) -> None:
    calls: list[tuple[tuple[int, ...], int]] = []

    class EndTurnPolicy:
        def choose_index(self, observation, action_mask, rng):
            calls.append((observation.shape, int(action_mask.sum())))
            return int(np.flatnonzero(action_mask)[-1])  # end turn is the last index

    env = BattlegroundsEnv(*content, opponents={"player-2": EndTurnPolicy()})
    env.reset(seed=4)
    env.step(env.action_layout.end_turn)

    assert calls and calls[0][0] == (env.observation_size,)
    assert env.engine.player_state("player-2").minions == ()


def test_configuration_errors(content) -> None:
    with pytest.raises(ValueError, match="learner seat cannot"):
        BattlegroundsEnv(*content, opponents={"player-1": object()})
    with pytest.raises(ValueError, match="not in this match"):
        BattlegroundsEnv(*content, learner_id="nobody").reset(seed=0)


def test_placement_reward_spans_plus_to_minus_one() -> None:
    assert placement_reward(1, 8) == 1.0
    assert placement_reward(8, 8) == -1.0
    assert placement_reward(4, 8) == pytest.approx(1 / 7)
    assert placement_reward(1, 1) == 0.0

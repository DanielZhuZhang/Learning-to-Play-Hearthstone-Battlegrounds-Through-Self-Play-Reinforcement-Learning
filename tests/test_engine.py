from dataclasses import replace
from pathlib import Path

import pytest

from hsbg.data.profiles import load_game_data
from hsbg.data.models import TokenSpec
from hsbg.game.actions import InvalidActionError
from hsbg.game.engine import GameEngine
from hsbg.game.legal import RecruitAction, RecruitActionKind
from hsbg.game.state import MatchPhase, MinionInstance, MinionZone

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _content():
    profile, cards = load_game_data(PROJECT_ROOT / "data" / "profiles" / "initial-15.6.json")
    return profile, cards


def test_engine_applies_legal_buy_and_play_with_battlecry_resolution() -> None:
    profile, cards = _content()
    engine = GameEngine(profile, cards, seed=8)
    player = engine.state.players[0]
    buy_action = next(
        action
        for action in engine.legal_actions(player.player_id)
        if action.kind is RecruitActionKind.BUY
    )

    engine.apply_action(player.player_id, buy_action)

    owned = engine.state.players[0].hand[0]
    assert owned.instance_id == f"{player.player_id}:minion:1"
    play_action = RecruitAction(RecruitActionKind.PLAY, owned.instance_id)
    engine.apply_action(player.player_id, play_action)

    updated = engine.state.players[0]
    assert owned.instance_id in {minion.instance_id for minion in updated.board}
    assert updated.gold == profile.rules.starting_gold - profile.rules.buy_cost
    assert len(engine.action_history) == 2


def test_ending_all_recruit_phases_advances_turn_and_refills_shops() -> None:
    profile, cards = _content()
    engine = GameEngine(profile, cards, seed=12)

    for player in engine.state.players:
        engine.apply_action(
            player.player_id,
            RecruitAction(RecruitActionKind.END_RECRUIT),
        )

    assert engine.state.turn == 2
    assert engine.state.phase is MatchPhase.RECRUIT
    assert all(player.gold == 4 for player in engine.state.players)
    assert all(len(player.shop) == 3 for player in engine.state.players)
    assert engine.pool.available_copies == engine.pool.total_copies - 24
    assert len(engine.combat_history) == 4


def test_replaying_ordered_actions_reproduces_match_state_and_combats() -> None:
    profile, cards = _content()
    engine = GameEngine(profile, cards, seed=99)

    for player in engine.state.players:
        engine.apply_action(
            player.player_id,
            RecruitAction(RecruitActionKind.END_RECRUIT),
        )

    replay = engine.to_replay()
    repeated = GameEngine.replay(profile, cards, replay)

    assert repeated.state == engine.state
    assert repeated.pool == engine.pool
    assert repeated.action_history == engine.action_history
    assert repeated.combat_history == engine.combat_history
    assert replay.seed == 99


def test_engine_rejects_action_not_in_current_legal_catalog() -> None:
    profile, cards = _content()
    engine = GameEngine(profile, cards, seed=101)

    with pytest.raises(InvalidActionError, match="not currently legal"):
        engine.apply_action(
            engine.state.players[0].player_id,
            RecruitAction(RecruitActionKind.BUY, "not-an-offer"),
        )


def test_game_engine_does_not_share_rng_objects_between_runs() -> None:
    profile, cards = _content()
    first = GameEngine(profile, cards, seed=44)
    second = GameEngine(profile, cards, seed=44)

    assert first.state == second.state
    assert first.rng is not second.rng
    assert first.rng.getstate() == second.rng.getstate()


def test_replay_rejects_changed_effective_rules_or_card_data() -> None:
    profile, cards = _content()
    engine = GameEngine(profile, cards, seed=103)
    replay = engine.to_replay()
    different_rules_profile = replace(
        profile,
        rules=replace(profile.rules, starting_health=profile.rules.starting_health + 1),
    )

    with pytest.raises(ValueError, match="profile or effective rules"):
        GameEngine.replay(different_rules_profile, cards, replay)
    changed_card = replace(cards[0], attack=cards[0].attack + 1)
    with pytest.raises(ValueError, match="card data"):
        GameEngine.replay(profile, (changed_card, *cards[1:]), replay)


def test_odd_lobby_adds_a_ghost_combat() -> None:
    profile, cards = _content()
    odd_profile = replace(
        profile,
        rules=replace(
            profile.rules,
            player_count=5,
            max_match_turns=1,
        ),
    )
    engine = GameEngine(odd_profile, cards, seed=107)

    for player in engine.state.players:
        engine.apply_action(
            player.player_id,
            RecruitAction(RecruitActionKind.END_RECRUIT),
        )

    assert engine.finished
    assert len(engine.combat_history) == 3
    assert any(
        player_id.startswith("ghost:")
        for result in engine.combat_history
        for player_id, _ in result.survivors_by_player
    )


def test_simultaneous_combat_eliminations_receive_a_shared_placement() -> None:
    profile, cards = _content()
    short_profile = replace(
        profile,
        rules=replace(
            profile.rules,
            player_count=4,
            starting_health=1,
            max_match_turns=1,
        ),
    )
    engine = GameEngine(short_profile, cards, seed=109)
    sizes = (0, 1, 2, 3)
    configured_players = []
    for player, board_size in zip(engine.state.players, sizes, strict=True):
        board = tuple(
            MinionInstance(
                instance_id=f"{player.player_id}:test-token:{index}",
                definition_id="test:token",
                zone=MinionZone.BOARD,
                pool_owned=False,
                token_spec=TokenSpec("Test Token", attack=20, health=100),
            )
            for index in range(board_size)
        )
        configured_players.append(replace(player, minions=board))
    engine.state = replace(engine.state, players=tuple(configured_players))

    for player in engine.state.players:
        engine.apply_action(
            player.player_id,
            RecruitAction(RecruitActionKind.END_RECRUIT),
        )

    eliminated = [player for player in engine.state.players if player.health <= 0]
    eliminated_places = {
        place
        for player_id, place in engine.placements
        if any(player.player_id == player_id for player in eliminated)
    }
    assert len(eliminated) == 2
    assert len(eliminated_places) == 1


def test_targeted_battlecry_offers_one_play_per_target_and_buffs_only_it() -> None:
    profile, cards = _content()
    engine = GameEngine(profile, cards, seed=4)
    player = engine.state.players[0]
    custom = replace(
        player,
        minions=(
            MinionInstance("demon-1", "15.6:3", MinionZone.BOARD),
            MinionInstance("beast-1", "15.6:1", MinionZone.BOARD),
            MinionInstance("demon-2", "15.6:3", MinionZone.BOARD),
            MinionInstance("overseer", "15.6:17", MinionZone.HAND),
        ),
    )
    engine.state = replace(
        engine.state, players=(custom,) + engine.state.players[1:]
    )

    plays = [
        action
        for action in engine.legal_actions(player.player_id)
        if action.kind is RecruitActionKind.PLAY
    ]
    assert plays == [
        RecruitAction(RecruitActionKind.PLAY, "overseer", "demon-1"),
        RecruitAction(RecruitActionKind.PLAY, "overseer", "demon-2"),
    ]
    with pytest.raises(InvalidActionError):
        engine.apply_action(
            player.player_id, RecruitAction(RecruitActionKind.PLAY, "overseer")
        )

    engine.apply_action(player.player_id, plays[1])

    board = {minion.instance_id: minion for minion in engine.state.players[0].board}
    assert (board["demon-2"].attack_bonus, board["demon-2"].health_bonus) == (2, 2)
    assert board["demon-1"].attack_bonus == 0
    assert board["beast-1"].attack_bonus == 0
    assert board["overseer"].attack_bonus == 0
    assert engine.action_history[-1].action.effect_target_id == "demon-2"

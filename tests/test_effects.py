from pathlib import Path

import pytest

from hsbg.data.profiles import load_profile
from hsbg.data.selection import select_enabled_cards
from hsbg.data.workbook import load_sheet
from hsbg.game.effects import (
    EffectResolutionError,
    battlecry_target_options,
    play_minion_and_resolve_battlecry,
    resolve_battlecry,
)
from hsbg.game.state import MinionInstance, MinionZone, PlayerState

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _profile_content():
    profile = load_profile(PROJECT_ROOT / "data" / "profiles" / "initial-15.6.json")
    cards = select_enabled_cards(
        profile,
        load_sheet(PROJECT_ROOT / profile.workbook, profile.sheet),
    )
    return profile, cards


def _player(*minions: MinionInstance, health: int = 30, armor: int = 0) -> PlayerState:
    return PlayerState(
        player_id="player-1",
        health=health,
        armor=armor,
        gold=3,
        tavern_tier=1,
        minions=tuple(minions),
    )


def test_alleycat_battlecry_summons_a_non_pool_token_when_played() -> None:
    profile, cards = _profile_content()
    player = _player(MinionInstance("alleycat-1", "15.6:1", MinionZone.HAND))

    updated = play_minion_and_resolve_battlecry(
        player,
        "alleycat-1",
        cards,
        profile.card_effects,
        profile.rules,
    )

    assert len(updated.board) == 2
    token = updated.board[1]
    assert not token.pool_owned
    assert token.token_spec is not None
    assert (
        token.token_spec.name,
        token.token_spec.attack,
        token.token_spec.health,
    ) == (
        "Cat",
        1,
        1,
    )
    assert player.hand[0].instance_id == "alleycat-1"


def test_battlecry_summon_respects_board_limit() -> None:
    profile, cards = _profile_content()
    minions = tuple(
        MinionInstance(f"board-{index}", "15.6:3", MinionZone.BOARD)
        for index in range(profile.rules.board_limit - 1)
    ) + (MinionInstance("alleycat-1", "15.6:1", MinionZone.HAND),)

    updated = play_minion_and_resolve_battlecry(
        _player(*minions),
        "alleycat-1",
        cards,
        profile.card_effects,
        profile.rules,
    )

    assert len(updated.board) == profile.rules.board_limit
    assert all(not minion.pool_owned for minion in updated.board if minion.token_spec)


def test_battlecry_damage_consumes_armor_before_health() -> None:
    profile, cards = _profile_content()
    player = _player(
        MinionInstance("whelp-smuggler-1", "15.6:4", MinionZone.BOARD),
        health=30,
        armor=1,
    )

    updated = resolve_battlecry(
        player,
        "whelp-smuggler-1",
        cards,
        profile.card_effects,
        profile.rules,
    )

    assert updated.armor == 0
    assert updated.health == 29


def test_battlecry_buff_only_affects_selected_matching_tribe() -> None:
    profile, cards = _profile_content()
    source = MinionInstance("battlecry-source", "15.6:10", MinionZone.BOARD)
    murloc = MinionInstance("murloc-target", "15.6:9", MinionZone.BOARD)
    beast = MinionInstance("beast-target", "15.6:1", MinionZone.BOARD)
    player = _player(source, murloc, beast)

    updated = resolve_battlecry(
        player,
        source.instance_id,
        cards,
        profile.card_effects,
        profile.rules,
        murloc.instance_id,
    )

    assert updated.minions[1].attack_bonus > murloc.attack_bonus
    assert updated.minions[0] == source
    assert updated.minions[2] == beast
    assert player.minions == (source, murloc, beast)


def test_targeted_battlecry_buffs_only_the_chosen_minion() -> None:
    profile, cards = _profile_content()
    first = MinionInstance("demon-1", "15.6:3", MinionZone.BOARD)
    second = MinionInstance("demon-2", "15.6:3", MinionZone.BOARD)
    overseer = MinionInstance("overseer", "15.6:17", MinionZone.HAND)
    player = _player(first, second, overseer)

    assert battlecry_target_options(
        player, "overseer", cards, profile.card_effects
    ) == ("demon-1", "demon-2")
    updated = play_minion_and_resolve_battlecry(
        player, "overseer", cards, profile.card_effects, profile.rules, "demon-2"
    )

    by_id = {minion.instance_id: minion for minion in updated.minions}
    assert (by_id["demon-2"].attack_bonus, by_id["demon-2"].health_bonus) == (2, 2)
    assert by_id["demon-1"] == first
    assert (by_id["overseer"].attack_bonus, by_id["overseer"].health_bonus) == (0, 0)


def test_targeted_battlecry_without_valid_target_does_nothing_to_itself() -> None:
    profile, cards = _profile_content()
    beast = MinionInstance("beast", "15.6:1", MinionZone.BOARD)
    overseer = MinionInstance("overseer", "15.6:17", MinionZone.HAND)
    player = _player(beast, overseer)

    assert battlecry_target_options(
        player, "overseer", cards, profile.card_effects
    ) == ()
    updated = play_minion_and_resolve_battlecry(
        player, "overseer", cards, profile.card_effects, profile.rules
    )

    assert all(
        (minion.attack_bonus, minion.health_bonus) == (0, 0)
        for minion in updated.minions
    )


def test_targeted_battlecry_rejects_missing_or_invalid_targets() -> None:
    profile, cards = _profile_content()
    demon = MinionInstance("demon", "15.6:3", MinionZone.BOARD)
    beast = MinionInstance("beast", "15.6:1", MinionZone.BOARD)
    overseer = MinionInstance("overseer", "15.6:17", MinionZone.HAND)
    player = _player(demon, beast, overseer)

    for target in (None, "beast", "overseer"):
        with pytest.raises(EffectResolutionError):
            play_minion_and_resolve_battlecry(
                player, "overseer", cards, profile.card_effects, profile.rules, target
            )


def test_untargeted_battlecry_rejects_a_target() -> None:
    profile, cards = _profile_content()
    demon = MinionInstance("demon", "15.6:3", MinionZone.BOARD)
    crystalweaver = MinionInstance("weaver", "15.6:32", MinionZone.HAND)
    player = _player(demon, crystalweaver)

    assert battlecry_target_options(
        player, "weaver", cards, profile.card_effects
    ) is None
    with pytest.raises(EffectResolutionError):
        play_minion_and_resolve_battlecry(
            player, "weaver", cards, profile.card_effects, profile.rules, "demon"
        )


def test_taunt_target_selector_reads_passive_profile_keywords() -> None:
    profile, cards = _profile_content()
    source = MinionInstance("buff-source", "15.6:70", MinionZone.BOARD)
    taunt = MinionInstance("taunt-target", "15.6:3", MinionZone.BOARD)
    non_taunt = MinionInstance("plain-target", "15.6:1", MinionZone.BOARD)
    player = _player(source, taunt, non_taunt)

    updated = resolve_battlecry(
        player,
        source.instance_id,
        cards,
        profile.card_effects,
        profile.rules,
    )

    assert updated.minions[1].attack_bonus > 0
    assert updated.minions[2] == non_taunt

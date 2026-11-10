import random

from hsbg.data.models import (
    BuffEffect,
    CardEffect,
    CardEffectBinding,
    CardRecord,
    EffectTarget,
    EffectTrigger,
    KeywordEffect,
    MinionKeyword,
    SummonEffect,
    TokenSpec,
)
from hsbg.game.combat import resolve_combat
from hsbg.game.config import GameRules
from hsbg.game.state import MinionInstance, MinionZone, PlayerState


def _card(
    card_id: str,
    *,
    attack: int = 1,
    health: int = 1,
    tribe: str | None = None,
    tier: int = 1,
) -> CardRecord:
    return CardRecord(
        card_id=card_id,
        tier=tier,
        name=card_id,
        tribe=tribe,
        card_type=tribe,
        attack=attack,
        health=health,
        text="",
    )


def _player(player_id: str, *card_ids: str, tier: int = 1) -> PlayerState:
    instances: list[MinionInstance] = []
    counts: dict[str, int] = {}
    for card_id in card_ids:
        counts[card_id] = counts.get(card_id, 0) + 1
        suffix = f"-{counts[card_id]}" if counts[card_id] > 1 else ""
        instances.append(
            MinionInstance(f"{player_id}-{card_id}{suffix}", card_id, MinionZone.BOARD)
        )
    return PlayerState(
        player_id=player_id,
        health=30,
        armor=0,
        gold=3,
        tavern_tier=tier,
        minions=tuple(instances),
    )


def _passive(card_id: str, keyword: MinionKeyword) -> CardEffectBinding:
    return CardEffectBinding(
        card_id,
        (
            CardEffect(
                EffectTrigger.PASSIVE,
                KeywordEffect(EffectTarget.SELF, keyword),
            ),
        ),
    )


def test_combat_forces_attacks_into_taunt_before_other_minions() -> None:
    cards = (
        _card("attacker-1", attack=2, health=3),
        _card("attacker-2", attack=2, health=3),
        _card("attacker-3", attack=2, health=3),
        _card("taunt", attack=1, health=3),
        _card("plain", attack=1, health=3),
    )
    first = _player("first", "attacker-1", "attacker-2", "attacker-3")
    second = _player("second", "taunt", "plain")

    result = resolve_combat(
        first,
        second,
        cards,
        (_passive("taunt", MinionKeyword.TAUNT),),
        random.Random(10),
        GameRules(),
    )

    first_attack = next(event for event in result.events if event.kind == "attack")
    assert first_attack.source_instance_id == "first-attacker-1"
    assert first_attack.target_instance_id == "second-taunt"


def test_divine_shield_absorbs_damage_before_health() -> None:
    cards = (
        _card("small-1", attack=1, health=2),
        _card("small-2", attack=1, health=2),
        _card("shielded", attack=1, health=3),
    )
    result = resolve_combat(
        _player("first", "small-1", "small-2"),
        _player("second", "shielded"),
        cards,
        (_passive("shielded", MinionKeyword.DIVINE_SHIELD),),
        random.Random(4),
        GameRules(),
    )

    assert any(event.kind == "shield" for event in result.events)


def test_poisonous_kills_a_larger_unshielded_target() -> None:
    cards = (
        _card("poison", attack=1, health=1),
        _card("giant", attack=8, health=20),
    )
    result = resolve_combat(
        _player("first", "poison", "poison"),
        _player("second", "giant"),
        cards,
        (_passive("poison", MinionKeyword.POISONOUS),),
        random.Random(3),
        GameRules(),
    )

    assert any(event.kind == "poison" for event in result.events)
    assert not result.survivors_by_player[1][1]


def test_deathrattle_summons_token_and_token_contributes_to_survivor_damage() -> None:
    cards = (
        _card("killer", attack=5, health=5, tier=2),
        _card("deathrattle", attack=1, health=1, tier=1),
        _card("solid", attack=10, health=10, tier=1),
    )
    deathrattle = CardEffectBinding(
        "deathrattle",
        (
            CardEffect(
                EffectTrigger.DEATHRATTLE,
                SummonEffect(TokenSpec("Rat", 1, 20), count=1),
            ),
        ),
    )
    result = resolve_combat(
        _player("first", "deathrattle", "solid", tier=2),
        _player("second", "killer"),
        cards,
        (deathrattle,),
        random.Random(5),
        GameRules(),
    )

    assert result.winner_id == "first"
    assert result.damage == 4
    assert any(event.kind == "summon" for event in result.events)
    assert any(minion.token for minion in result.survivors_by_player[0][1])


def test_deathrattle_buff_is_temporary_to_combat_state() -> None:
    cards = (
        _card("killer", attack=5, health=5),
        _card("rover", attack=1, health=1),
        _card("friend", attack=20, health=20),
    )
    deathrattle = CardEffectBinding(
        "rover",
        (
            CardEffect(
                EffectTrigger.DEATHRATTLE,
                BuffEffect(EffectTarget.ALL_FRIENDLY_MINIONS, attack=2, health=1),
            ),
        ),
    )
    first = _player("first", "killer")
    second = _player("second", "rover", "friend")

    result = resolve_combat(
        first,
        second,
        cards,
        (deathrattle,),
        random.Random(6),
        GameRules(),
    )

    friend = next(
        minion
        for minion in result.survivors_by_player[1][1]
        if minion.instance_id == "second-friend"
    )
    assert friend.attack == 22
    assert friend.health == 16
    assert friend.max_health == 21
    assert second.board[1].attack_bonus == 0
    assert first.health == 30


def test_recorded_frames_do_not_change_the_outcome_and_end_at_survivors() -> None:
    cards = (
        _card("egg", attack=1, health=3),
        _card("brute", attack=3, health=2),
    )
    bindings = (
        CardEffectBinding(
            "egg",
            (
                CardEffect(
                    EffectTrigger.DEATHRATTLE,
                    SummonEffect(TokenSpec("Hatchling", 2, 2), 1),
                ),
            ),
        ),
    )
    first = _player("first", "egg", "egg")
    second = _player("second", "brute", "brute")

    plain = resolve_combat(first, second, cards, bindings, random.Random(5), GameRules())
    recorded = resolve_combat(
        first,
        second,
        cards,
        bindings,
        random.Random(5),
        GameRules(),
        record_frames=True,
    )

    assert plain.frames == ()
    assert recorded.events == plain.events
    assert recorded.survivors_by_player == plain.survivors_by_player
    assert recorded.frames[0].kind == "start"
    assert recorded.frames[-1].kind == "end"
    assert {"attack", "hit", "deaths", "deathrattle"} <= {
        frame.kind for frame in recorded.frames
    }
    final_boards = tuple(survivors for _player_id, survivors in recorded.survivors_by_player)
    assert recorded.frames[-1].boards == final_boards
    summoned = [
        minion
        for frame in recorded.frames
        for board in frame.boards
        for minion in board
        if minion.token
    ]
    assert summoned and all(minion.token_name == "Hatchling" for minion in summoned)

from dataclasses import FrozenInstanceError

import pytest

from hsbg.data.models import MinionKeyword
from hsbg.game.state import (
    MatchPhase,
    MatchState,
    MinionInstance,
    MinionZone,
    PlayerState,
    ShopOffer,
)


def _player(
    *,
    player_id: str = "player-1",
    minions: tuple[MinionInstance, ...] = (),
    shop: tuple[ShopOffer, ...] = (),
) -> PlayerState:
    return PlayerState(
        player_id=player_id,
        health=40,
        armor=0,
        gold=3,
        tavern_tier=1,
        minions=minions,
        shop=shop,
    )


def test_player_zone_views_distinguish_hand_and_board_instances() -> None:
    hand_minion = MinionInstance("instance-1", "15.6:1", MinionZone.HAND)
    board_minion = MinionInstance(
        "instance-2", "15.6:1", MinionZone.BOARD, attack_bonus=2
    )
    player = _player(minions=(hand_minion, board_minion))

    assert player.hand == (hand_minion,)
    assert player.board == (board_minion,)
    assert player.hand[0].zone is MinionZone.HAND
    assert player.board[0].zone is MinionZone.BOARD


def test_shop_offer_and_owned_instance_are_separate_values() -> None:
    offer = ShopOffer(
        "offer-1",
        "15.6:1",
        attack_bonus=1,
        health_bonus=2,
        granted_keywords=(MinionKeyword.TAUNT,),
    )
    purchased_copy = MinionInstance(
        "instance-1",
        offer.definition_id,
        MinionZone.HAND,
        attack_bonus=offer.attack_bonus,
        health_bonus=offer.health_bonus,
        granted_keywords=offer.granted_keywords,
    )
    player = _player(shop=(offer,), minions=(purchased_copy,))

    assert player.shop == (offer,)
    assert player.hand == (purchased_copy,)
    assert player.shop[0].offer_id != player.hand[0].instance_id
    assert player.shop[0].definition_id == player.hand[0].definition_id
    assert player.shop[0].attack_bonus == player.hand[0].attack_bonus

    with pytest.raises(FrozenInstanceError):
        offer.attack_bonus = 5  # type: ignore[misc]


def test_freezing_shop_does_not_change_offer_or_owned_copy() -> None:
    offer = ShopOffer("offer-1", "15.6:1", attack_bonus=2)
    copy = MinionInstance("instance-1", "15.6:1", MinionZone.HAND, attack_bonus=2)

    player = PlayerState(
        player_id="player-1",
        health=40,
        armor=0,
        gold=3,
        tavern_tier=1,
        minions=(copy,),
        shop=(offer,),
        shop_frozen=True,
    )

    assert player.shop_frozen
    assert player.shop == (offer,)
    assert player.hand == (copy,)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"player_id": ""}, "Player ID must not be empty"),
        ({"health": -1}, "health must be a nonnegative integer"),
        ({"armor": -1}, "armor must be a nonnegative integer"),
        ({"gold": -1}, "gold must be a nonnegative integer"),
        ({"tavern_tier": 0}, "tavern_tier must be a positive integer"),
        ({"shop_frozen": 1}, "shop_frozen must be a boolean"),
    ],
)
def test_player_state_rejects_invalid_values(
    changes: dict[str, object], message: str
) -> None:
    values: dict[str, object] = {
        "player_id": "player-1",
        "health": 40,
        "armor": 0,
        "gold": 3,
        "tavern_tier": 1,
    }
    values.update(changes)

    with pytest.raises(ValueError, match=message):
        PlayerState(**values)  # type: ignore[arg-type]


def test_player_state_rejects_duplicate_owned_instance_ids() -> None:
    duplicate = MinionInstance("instance-1", "15.6:1", MinionZone.HAND)

    with pytest.raises(ValueError, match="instance IDs must be unique"):
        _player(minions=(duplicate, duplicate))


def test_player_state_rejects_duplicate_shop_offer_ids() -> None:
    duplicate = ShopOffer("offer-1", "15.6:1")

    with pytest.raises(ValueError, match="Shop offer IDs must be unique"):
        _player(shop=(duplicate, duplicate))


def test_shop_offer_and_minion_instance_require_identifiers_and_valid_buffs() -> None:
    with pytest.raises(ValueError, match="Shop offer ID"):
        ShopOffer("", "15.6:1")

    with pytest.raises(ValueError, match="definition ID"):
        MinionInstance("instance-1", "", MinionZone.HAND)

    with pytest.raises(ValueError, match="attack_bonus must be a nonnegative integer"):
        ShopOffer("offer-1", "15.6:1", attack_bonus=-1)


def test_match_state_validates_turn_and_unique_players() -> None:
    player = _player()
    other_player = _player(player_id="player-2")
    match = MatchState(turn=1, phase=MatchPhase.RECRUIT, players=(player, other_player))

    assert match.phase is MatchPhase.RECRUIT

    with pytest.raises(ValueError, match="turn must be a positive integer"):
        MatchState(turn=0, phase=MatchPhase.RECRUIT, players=(player,))

    with pytest.raises(ValueError, match="Player IDs must be unique"):
        MatchState(
            turn=1,
            phase=MatchPhase.COMBAT,
            players=(player, player),
        )

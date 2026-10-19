import random
from dataclasses import replace

import pytest

from hsbg.data.models import MinionKeyword
from hsbg.game.actions import (
    InvalidActionError,
    buy_offer,
    play_minion,
    reorder_board,
    reroll_shop,
    sell_minion,
    set_shop_frozen,
    start_recruit_turn,
    upgrade_tavern,
)
from hsbg.game.config import GameRules
from hsbg.game.pool import MinionPool, MinionPoolEntry, PoolAccountingError
from hsbg.game.state import MinionInstance, MinionZone, PlayerState, ShopOffer


def _player(
    *,
    gold: int = 3,
    minions: tuple[MinionInstance, ...] = (),
    shop: tuple[ShopOffer, ...] | None = None,
    shop_frozen: bool = False,
) -> PlayerState:
    return PlayerState(
        player_id="player-1",
        health=40,
        armor=0,
        gold=gold,
        tavern_tier=1,
        minions=minions,
        shop=shop
        if shop is not None
        else (ShopOffer("offer-1", "15.6:1"), ShopOffer("offer-2", "15.6:3")),
        shop_frozen=shop_frozen,
    )


def test_buy_offer_creates_a_hand_instance_and_preserves_other_state() -> None:
    offer = ShopOffer(
        "offer-1",
        "15.6:1",
        attack_bonus=2,
        health_bonus=1,
        granted_keywords=(MinionKeyword.TAUNT,),
    )
    player = _player(shop=(offer, ShopOffer("offer-2", "15.6:3")), shop_frozen=True)

    updated = buy_offer(player, "offer-1", "instance-1", GameRules())

    assert updated.gold == 0
    assert updated.shop == (ShopOffer("offer-2", "15.6:3"),)
    assert updated.shop_frozen
    assert updated.board == ()
    assert len(updated.hand) == 1
    assert updated.hand[0] == MinionInstance(
        instance_id="instance-1",
        definition_id="15.6:1",
        zone=MinionZone.HAND,
        attack_bonus=2,
        health_bonus=1,
        granted_keywords=(MinionKeyword.TAUNT,),
    )
    assert player.gold == 3
    assert player.shop[0] == offer
    assert player.minions == ()


@pytest.mark.parametrize(
    ("player", "offer_id", "instance_id", "rules", "message"),
    [
        (_player(), "missing", "instance-1", GameRules(), "does not exist"),
        (_player(gold=2), "offer-1", "instance-1", GameRules(), "costs 3 gold"),
        (
            _player(minions=(MinionInstance("held", "15.6:2", MinionZone.HAND),)),
            "offer-1",
            "instance-1",
            replace(GameRules(), hand_limit=1),
            "Hand is full",
        ),
        (
            _player(
                minions=(MinionInstance("instance-1", "15.6:2", MinionZone.BOARD),)
            ),
            "offer-1",
            "instance-1",
            GameRules(),
            "already exists",
        ),
    ],
)
def test_buy_offer_rejects_illegal_purchases(
    player: PlayerState,
    offer_id: str,
    instance_id: str,
    rules: GameRules,
    message: str,
) -> None:
    with pytest.raises(InvalidActionError, match=message):
        buy_offer(player, offer_id, instance_id, rules)


def test_buy_offer_does_not_execute_battlecry() -> None:
    player = _player(shop=(ShopOffer("offer-1", "15.6:1"),))

    updated = buy_offer(player, "offer-1", "instance-1", GameRules())

    assert updated.hand == (MinionInstance("instance-1", "15.6:1", MinionZone.HAND),)


def test_play_minion_moves_instance_to_board_and_preserves_modifiers() -> None:
    on_board = MinionInstance("board-1", "15.6:2", MinionZone.BOARD)
    in_hand = MinionInstance(
        "hand-1",
        "15.6:1",
        MinionZone.HAND,
        attack_bonus=2,
        health_bonus=3,
        granted_keywords=(MinionKeyword.TAUNT,),
    )
    other_in_hand = MinionInstance("hand-2", "15.6:3", MinionZone.HAND)
    player = _player(minions=(on_board, in_hand, other_in_hand), shop=())

    updated = play_minion(player, "hand-1", GameRules())

    assert updated.board == (
        on_board,
        replace(in_hand, zone=MinionZone.BOARD),
    )
    assert updated.hand == (other_in_hand,)
    assert player.minions == (on_board, in_hand, other_in_hand)


@pytest.mark.parametrize(
    ("player", "instance_id", "rules", "message"),
    [
        (_player(shop=()), "missing", GameRules(), "does not exist"),
        (
            _player(
                minions=(MinionInstance("board-1", "15.6:1", MinionZone.BOARD),),
                shop=(),
            ),
            "board-1",
            GameRules(),
            "not in the player's hand",
        ),
        (
            _player(
                minions=(
                    MinionInstance("board-1", "15.6:1", MinionZone.BOARD),
                    MinionInstance("hand-1", "15.6:2", MinionZone.HAND),
                ),
                shop=(),
            ),
            "hand-1",
            replace(GameRules(), board_limit=1),
            "Board is full",
        ),
    ],
)
def test_play_minion_rejects_invalid_transitions(
    player: PlayerState,
    instance_id: str,
    rules: GameRules,
    message: str,
) -> None:
    with pytest.raises(InvalidActionError, match=message):
        play_minion(player, instance_id, rules)


def test_set_shop_frozen_changes_only_freeze_state() -> None:
    player = _player(shop_frozen=False)

    frozen = set_shop_frozen(player, True)
    unfrozen = set_shop_frozen(frozen, False)

    assert frozen.shop_frozen
    assert not unfrozen.shop_frozen
    assert frozen.shop == player.shop
    assert unfrozen.gold == player.gold
    assert not player.shop_frozen


def test_set_shop_frozen_rejects_non_boolean_values() -> None:
    with pytest.raises(InvalidActionError, match="must be a boolean"):
        set_shop_frozen(_player(), 1)  # type: ignore[arg-type]


def test_upgrade_tavern_uses_next_tier_cost_and_preserves_other_state() -> None:
    player = replace(_player(gold=10), tavern_tier=2)

    upgraded = upgrade_tavern(player, GameRules())

    assert upgraded.gold == 3
    assert upgraded.tavern_tier == 3
    assert upgraded.shop == player.shop
    assert upgraded.minions == player.minions
    assert player.gold == 10
    assert player.tavern_tier == 2


@pytest.mark.parametrize(
    ("player", "rules", "message"),
    [
        (
            replace(_player(gold=6), tavern_tier=2),
            GameRules(),
            "to Tavern tier 3 costs 7 gold",
        ),
        (
            replace(_player(gold=10), tavern_tier=6),
            GameRules(),
            "already at the maximum Tavern tier",
        ),
        (
            replace(_player(gold=10), tavern_tier=7),
            GameRules(),
            "already at the maximum Tavern tier",
        ),
    ],
)
def test_upgrade_tavern_rejects_illegal_upgrades(
    player: PlayerState,
    rules: GameRules,
    message: str,
) -> None:
    with pytest.raises(InvalidActionError, match=message):
        upgrade_tavern(player, rules)


def test_start_recruit_turn_sets_gold_and_draws_when_not_frozen() -> None:
    player = _player(gold=0, shop=())
    pool = MinionPool(
        entries=(MinionPoolEntry("card-a", tier=1, total_copies=3, available_copies=3),)
    )

    updated_player, updated_pool = start_recruit_turn(
        player,
        pool,
        3,
        ("card-a",),
        ("offer-1", "offer-2", "offer-3"),
        random.Random(1),
        GameRules(),
    )

    assert updated_player.gold == 5
    assert updated_player.shop == (
        ShopOffer("offer-1", "card-a"),
        ShopOffer("offer-2", "card-a"),
        ShopOffer("offer-3", "card-a"),
    )
    assert updated_pool.available_copies == 0
    assert player.gold == 0
    assert player.shop == ()
    assert pool.available_copies == 3


def test_start_recruit_turn_retains_frozen_shop_and_caps_gold() -> None:
    player = _player(
        gold=1,
        shop=(ShopOffer("offer-1", "card-a"),),
        shop_frozen=True,
    )
    pool = MinionPool(
        entries=(MinionPoolEntry("card-a", tier=1, total_copies=1, available_copies=0),)
    )

    updated_player, updated_pool = start_recruit_turn(
        player,
        pool,
        20,
        (),
        (),
        random.Random(1),
        GameRules(),
    )

    assert updated_player.gold == 10
    assert updated_player.shop == player.shop
    assert not updated_player.shop_frozen
    assert updated_pool is pool
    assert pool.available_copies == 0


def test_start_recruit_turn_returns_old_offers_before_refill() -> None:
    player = _player(gold=0, shop=(ShopOffer("old-offer", "card-a"),))
    pool = MinionPool(
        entries=(
            MinionPoolEntry("card-a", tier=1, total_copies=1, available_copies=0),
            MinionPoolEntry("card-b", tier=1, total_copies=1, available_copies=1),
        )
    )

    updated_player, updated_pool = start_recruit_turn(
        player,
        pool,
        2,
        ("card-b",),
        ("new-offer-1", "new-offer-2", "new-offer-3"),
        random.Random(1),
        GameRules(),
    )

    assert updated_player.gold == 4
    assert updated_player.shop == (ShopOffer("new-offer-1", "card-b"),)
    assert updated_pool.entries[0].available_copies == 1
    assert updated_pool.entries[1].available_copies == 0


@pytest.mark.parametrize("turn", [0, -1, True, 1.5])
def test_start_recruit_turn_rejects_invalid_turn(turn: object) -> None:
    with pytest.raises(InvalidActionError, match="positive integer"):
        start_recruit_turn(
            _player(),
            MinionPool(entries=()),
            turn,  # type: ignore[arg-type]
            (),
            (),
            random.Random(1),
            GameRules(),
        )


def test_sell_minion_returns_pool_copy_and_refunds_gold() -> None:
    minion = MinionInstance("board-1", "card-a", MinionZone.BOARD)
    player = _player(gold=2, minions=(minion,), shop=())
    pool = MinionPool(
        entries=(MinionPoolEntry("card-a", tier=1, total_copies=1, available_copies=0),)
    )

    updated_player, updated_pool = sell_minion(player, pool, "board-1", GameRules())

    assert updated_player.gold == 3
    assert updated_player.minions == ()
    assert updated_pool.entries[0].available_copies == 1
    assert player.gold == 2
    assert player.minions == (minion,)
    assert pool.entries[0].available_copies == 0


def test_sell_minion_does_not_return_token_copy_and_caps_gold() -> None:
    token = MinionInstance("token-1", "token:cat", MinionZone.BOARD, pool_owned=False)
    player = _player(gold=10, minions=(token,), shop=())
    pool = MinionPool(
        entries=(MinionPoolEntry("card-a", tier=1, total_copies=1, available_copies=0),)
    )

    updated_player, updated_pool = sell_minion(player, pool, "token-1", GameRules())

    assert updated_player.gold == 10
    assert updated_player.minions == ()
    assert updated_pool is pool


@pytest.mark.parametrize(
    ("player", "instance_id", "message"),
    [
        (_player(shop=()), "missing", "does not exist"),
        (
            _player(
                minions=(MinionInstance("hand-1", "card-a", MinionZone.HAND),),
                shop=(),
            ),
            "hand-1",
            "not on the player's board",
        ),
    ],
)
def test_sell_minion_rejects_invalid_transitions(
    player: PlayerState,
    instance_id: str,
    message: str,
) -> None:
    pool = MinionPool(
        entries=(MinionPoolEntry("card-a", tier=1, total_copies=1, available_copies=0),)
    )

    with pytest.raises(InvalidActionError, match=message):
        sell_minion(player, pool, instance_id, GameRules())


def test_sell_minion_preserves_state_if_pool_return_is_invalid() -> None:
    player = _player(
        minions=(MinionInstance("board-1", "card-a", MinionZone.BOARD),),
        shop=(),
    )
    pool = MinionPool(
        entries=(MinionPoolEntry("card-a", tier=1, total_copies=1, available_copies=1),)
    )

    with pytest.raises(PoolAccountingError, match="exceed its configured"):
        sell_minion(player, pool, "board-1", GameRules())

    assert player.gold == 3
    assert len(player.minions) == 1
    assert pool.entries[0].available_copies == 1


def test_reorder_board_changes_board_order_and_preserves_hand_order() -> None:
    first_board = MinionInstance("board-1", "card-a", MinionZone.BOARD)
    hand = MinionInstance("hand-1", "card-b", MinionZone.HAND)
    second_board = MinionInstance("board-2", "card-c", MinionZone.BOARD)
    player = _player(minions=(first_board, hand, second_board), shop=())

    updated = reorder_board(player, ("board-2", "board-1"))

    assert updated.board == (second_board, first_board)
    assert updated.hand == (hand,)
    assert updated.minions == (second_board, hand, first_board)
    assert player.board == (first_board, second_board)


@pytest.mark.parametrize(
    ("instance_ids", "message"),
    [
        (("board-1", "board-1"), "duplicate instance IDs"),
        (("board-1",), "each current board minion exactly once"),
        (("board-1", "missing"), "each current board minion exactly once"),
        (("", "board-2"), "non-empty instance IDs"),
    ],
)
def test_reorder_board_rejects_invalid_orders(
    instance_ids: tuple[str, ...],
    message: str,
) -> None:
    player = _player(
        minions=(
            MinionInstance("board-1", "card-a", MinionZone.BOARD),
            MinionInstance("board-2", "card-b", MinionZone.BOARD),
        ),
        shop=(),
    )

    with pytest.raises(InvalidActionError, match=message):
        reorder_board(player, instance_ids)


def test_reroll_shop_returns_old_offers_and_draws_new_ones() -> None:
    player = _player(
        gold=3,
        shop=(ShopOffer("old-offer", "card-a"),),
        shop_frozen=True,
    )
    pool = MinionPool(
        entries=(
            MinionPoolEntry("card-a", tier=1, total_copies=1, available_copies=0),
            MinionPoolEntry("card-b", tier=1, total_copies=1, available_copies=1),
        )
    )

    updated_player, updated_pool = reroll_shop(
        player,
        pool,
        ("card-b",),
        ("new-offer-1", "new-offer-2", "new-offer-3"),
        random.Random(1),
        GameRules(),
    )

    assert updated_player.gold == 2
    assert updated_player.shop == (ShopOffer("new-offer-1", "card-b"),)
    assert not updated_player.shop_frozen
    assert updated_pool.entries[0].available_copies == 1
    assert updated_pool.entries[1].available_copies == 0
    assert player.gold == 3
    assert player.shop == (ShopOffer("old-offer", "card-a"),)
    assert pool.entries[0].available_copies == 0
    assert pool.entries[1].available_copies == 1


def test_reroll_shop_rejects_insufficient_gold_without_changing_inputs() -> None:
    player = _player(gold=0, shop=(ShopOffer("old-offer", "card-a"),))
    pool = MinionPool(
        entries=(MinionPoolEntry("card-a", tier=1, total_copies=1, available_copies=0),)
    )

    with pytest.raises(InvalidActionError, match="costs 1 gold"):
        reroll_shop(
            player,
            pool,
            ("card-a",),
            ("new-offer-1", "new-offer-2", "new-offer-3"),
            random.Random(1),
            GameRules(),
        )

    assert player.gold == 0
    assert player.shop == (ShopOffer("old-offer", "card-a"),)
    assert pool.entries[0].available_copies == 0


def test_reroll_shop_rejects_shop_offer_not_reserved_in_pool() -> None:
    player = _player(shop=(ShopOffer("old-offer", "card-a"),))
    pool = MinionPool(
        entries=(MinionPoolEntry("card-a", tier=1, total_copies=1, available_copies=1),)
    )

    with pytest.raises(PoolAccountingError, match="exceed its configured"):
        reroll_shop(
            player,
            pool,
            ("card-a",),
            ("new-offer-1", "new-offer-2", "new-offer-3"),
            random.Random(1),
            GameRules(),
        )

    assert player.gold == 3
    assert player.shop == (ShopOffer("old-offer", "card-a"),)
    assert pool.entries[0].available_copies == 1

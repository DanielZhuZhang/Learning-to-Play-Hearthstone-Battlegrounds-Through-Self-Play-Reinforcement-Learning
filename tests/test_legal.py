from dataclasses import replace

import pytest

from hsbg.game.config import GameRules
from hsbg.game.legal import (
    RecruitAction,
    RecruitActionKind,
    get_legal_recruit_actions,
)
from hsbg.game.state import MinionInstance, MinionZone, PlayerState, ShopOffer


def _player(
    *,
    gold: int = 3,
    tier: int = 1,
    minions: tuple[MinionInstance, ...] = (),
    shop: tuple[ShopOffer, ...] = (ShopOffer("offer-1", "card-1"),),
    frozen: bool = False,
) -> PlayerState:
    return PlayerState(
        player_id="player-1",
        health=30,
        armor=0,
        gold=gold,
        tavern_tier=tier,
        minions=minions,
        shop=shop,
        shop_frozen=frozen,
    )


def test_legal_actions_reflect_gold_and_populations_in_stable_order() -> None:
    player = _player(
        minions=(
            MinionInstance("hand-1", "card-1", MinionZone.HAND),
            MinionInstance("board-1", "card-2", MinionZone.BOARD),
            MinionInstance("board-2", "card-3", MinionZone.BOARD),
        )
    )

    actions = get_legal_recruit_actions(player, GameRules())

    assert actions == (
        RecruitAction(RecruitActionKind.BUY, "offer-1"),
        RecruitAction(RecruitActionKind.PLAY, "hand-1"),
        RecruitAction(RecruitActionKind.SELL, "board-1"),
        RecruitAction(RecruitActionKind.SELL, "board-2"),
        RecruitAction(RecruitActionKind.REROLL),
        RecruitAction(RecruitActionKind.FREEZE),
        RecruitAction(RecruitActionKind.REORDER_BOARD),
        RecruitAction(RecruitActionKind.END_RECRUIT),
    )


def test_legal_actions_omit_unaffordable_or_capacity_blocked_actions() -> None:
    rules = replace(GameRules(), hand_limit=1, board_limit=1)
    player = _player(
        gold=0,
        minions=(
            MinionInstance("hand-1", "card-1", MinionZone.HAND),
            MinionInstance("board-1", "card-2", MinionZone.BOARD),
        ),
    )

    actions = get_legal_recruit_actions(player, rules)

    assert actions == (
        RecruitAction(RecruitActionKind.SELL, "board-1"),
        RecruitAction(RecruitActionKind.FREEZE),
        RecruitAction(RecruitActionKind.END_RECRUIT),
    )


def test_legal_actions_offer_unfreeze_and_upgrade_when_available() -> None:
    player = _player(
        gold=10,
        tier=2,
        frozen=True,
        shop=(ShopOffer("offer-1", "card-1"),),
    )

    actions = get_legal_recruit_actions(player, GameRules())

    assert actions == (
        RecruitAction(RecruitActionKind.BUY, "offer-1"),
        RecruitAction(RecruitActionKind.REROLL),
        RecruitAction(RecruitActionKind.UNFREEZE),
        RecruitAction(RecruitActionKind.UPGRADE_TAVERN),
        RecruitAction(RecruitActionKind.END_RECRUIT),
    )


@pytest.mark.parametrize(
    ("kind", "target", "message"),
    [
        (RecruitActionKind.BUY, None, "requires a target ID"),
        (RecruitActionKind.SELL, "", "must not be empty"),
        (RecruitActionKind.REROLL, "offer-1", "does not accept a target ID"),
    ],
)
def test_recruit_action_validates_target_contract(
    kind: RecruitActionKind,
    target: str | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        RecruitAction(kind, target)

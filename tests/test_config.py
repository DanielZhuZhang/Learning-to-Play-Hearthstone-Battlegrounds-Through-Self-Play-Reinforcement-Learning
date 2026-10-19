from dataclasses import replace

import pytest

from hsbg.game.config import GameRules


def test_game_rules_defaults_match_approved_initial_values() -> None:
    assert GameRules() == GameRules(
        player_count=8,
        starting_health=30,
        starting_armor=0,
        starting_gold=3,
        gold_per_turn=1,
        max_gold=10,
        buy_cost=3,
        sell_refund=1,
        reroll_cost=1,
        starting_tavern_tier=1,
        max_tavern_tier=6,
        tavern_upgrade_costs=(5, 7, 8, 9, 10),
        shop_slots_by_tier=(3, 4, 5, 6, 7, 7),
        pool_copies_by_tier=(16, 15, 13, 11, 9, 7),
        board_limit=7,
        hand_limit=10,
        combat_damage_cap=15,
        rematch_avoidance_rounds=1,
        max_match_turns=100,
    )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"player_count": 1}, "player_count must be at least 2"),
        ({"starting_health": 0}, "starting_health must be positive"),
        ({"starting_armor": -1}, "starting_armor must not be negative"),
        ({"starting_gold": -1}, "starting_gold must not be negative"),
        ({"max_gold": 2}, "max_gold must be at least starting_gold"),
        ({"buy_cost": 0}, "buy_cost must be positive"),
        ({"sell_refund": -1}, "sell_refund must not be negative"),
        ({"reroll_cost": 0}, "reroll_cost must be positive"),
        ({"starting_tavern_tier": 0}, "starting_tavern_tier must be at least 1"),
        (
            {"starting_tavern_tier": 3, "max_tavern_tier": 2},
            "max_tavern_tier must be at least starting_tavern_tier",
        ),
        (
            {"tavern_upgrade_costs": (5, 7)},
            "tavern_upgrade_costs must contain one cost for each tier above 1",
        ),
        (
            {"tavern_upgrade_costs": (5, 7, 8, 9, 0)},
            "Upgrade cost for Tavern tier 6 must be positive",
        ),
        (
            {"tavern_upgrade_costs": (5, 7, 8, 9, True)},
            "Upgrade cost for Tavern tier 6 must be an integer",
        ),
        (
            {"pool_copies_by_tier": (16, 15)},
            "pool_copies_by_tier must contain one count for each Tavern tier",
        ),
        (
            {"pool_copies_by_tier": (16, 15, 13, 11, 9, 0)},
            "Pool copy count for Tavern tier 6 must be positive",
        ),
        (
            {"pool_copies_by_tier": (16, 15, 13, 11, 9, True)},
            "Pool copy count for Tavern tier 6 must be an integer",
        ),
        (
            {"shop_slots_by_tier": (3, 4)},
            "shop_slots_by_tier must contain one size for each Tavern tier",
        ),
        (
            {"shop_slots_by_tier": (3, 4, 5, 6, 7, 0)},
            "Shop slot count for Tavern tier 6 must be positive",
        ),
        ({"board_limit": 0}, "board_limit must be positive"),
        ({"hand_limit": 0}, "hand_limit must be positive"),
        ({"combat_damage_cap": 0}, "combat_damage_cap must be positive"),
        (
            {"rematch_avoidance_rounds": -1},
            "rematch_avoidance_rounds must not be negative",
        ),
        ({"max_match_turns": 0}, "max_match_turns must be positive"),
        ({"player_count": True}, "player_count must be an integer"),
    ],
)
def test_game_rules_reject_invalid_values(
    changes: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        replace(GameRules(), **changes)

"""Configurable match and recruit rules."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GameRules:
    """Initial match/recruit constants, independent of gameplay execution."""

    player_count: int = 8
    starting_health: int = 30
    starting_armor: int = 0
    starting_gold: int = 3
    gold_per_turn: int = 1
    max_gold: int = 10
    buy_cost: int = 3
    sell_refund: int = 1
    reroll_cost: int = 1
    starting_tavern_tier: int = 1
    max_tavern_tier: int = 6
    tavern_upgrade_costs: tuple[int, ...] = (5, 7, 8, 9, 10)
    shop_slots_by_tier: tuple[int, ...] = (3, 4, 5, 6, 7, 7)
    pool_copies_by_tier: tuple[int, ...] = (16, 15, 13, 11, 9, 7)
    board_limit: int = 7
    hand_limit: int = 10
    combat_damage_cap: int = 15
    rematch_avoidance_rounds: int = 1
    max_match_turns: int = 100

    def __post_init__(self) -> None:
        values = {
            "player_count": self.player_count,
            "starting_health": self.starting_health,
            "starting_armor": self.starting_armor,
            "starting_gold": self.starting_gold,
            "gold_per_turn": self.gold_per_turn,
            "max_gold": self.max_gold,
            "buy_cost": self.buy_cost,
            "sell_refund": self.sell_refund,
            "reroll_cost": self.reroll_cost,
            "starting_tavern_tier": self.starting_tavern_tier,
            "max_tavern_tier": self.max_tavern_tier,
            "board_limit": self.board_limit,
            "hand_limit": self.hand_limit,
            "combat_damage_cap": self.combat_damage_cap,
            "rematch_avoidance_rounds": self.rematch_avoidance_rounds,
            "max_match_turns": self.max_match_turns,
        }
        for name, value in values.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer.")

        if self.player_count < 2:
            raise ValueError("player_count must be at least 2.")
        if self.starting_health < 1:
            raise ValueError("starting_health must be positive.")
        if self.starting_armor < 0:
            raise ValueError("starting_armor must not be negative.")
        if self.starting_gold < 0:
            raise ValueError("starting_gold must not be negative.")
        if self.gold_per_turn < 0:
            raise ValueError("gold_per_turn must not be negative.")
        if self.max_gold < self.starting_gold:
            raise ValueError("max_gold must be at least starting_gold.")
        if self.buy_cost < 1:
            raise ValueError("buy_cost must be positive.")
        if self.sell_refund < 0:
            raise ValueError("sell_refund must not be negative.")
        if self.reroll_cost < 1:
            raise ValueError("reroll_cost must be positive.")
        if self.starting_tavern_tier < 1:
            raise ValueError("starting_tavern_tier must be at least 1.")
        if self.max_tavern_tier < self.starting_tavern_tier:
            raise ValueError("max_tavern_tier must be at least starting_tavern_tier.")
        if not isinstance(self.tavern_upgrade_costs, tuple):
            raise ValueError("tavern_upgrade_costs must be a tuple.")
        if len(self.tavern_upgrade_costs) != self.max_tavern_tier - 1:
            raise ValueError(
                "tavern_upgrade_costs must contain one cost for each tier above 1."
            )
        for target_tier, cost in enumerate(self.tavern_upgrade_costs, start=2):
            if isinstance(cost, bool) or not isinstance(cost, int):
                raise ValueError(
                    f"Upgrade cost for Tavern tier {target_tier} must be an integer."
                )
            if cost < 1:
                raise ValueError(
                    f"Upgrade cost for Tavern tier {target_tier} must be positive."
                )
        if not isinstance(self.shop_slots_by_tier, tuple):
            raise ValueError("shop_slots_by_tier must be a tuple.")
        if len(self.shop_slots_by_tier) != self.max_tavern_tier:
            raise ValueError(
                "shop_slots_by_tier must contain one size for each Tavern tier."
            )
        for tier, slots in enumerate(self.shop_slots_by_tier, start=1):
            if isinstance(slots, bool) or not isinstance(slots, int):
                raise ValueError(
                    f"Shop slot count for Tavern tier {tier} must be an integer."
                )
            if slots < 1:
                raise ValueError(
                    f"Shop slot count for Tavern tier {tier} must be positive."
                )
        if not isinstance(self.pool_copies_by_tier, tuple):
            raise ValueError("pool_copies_by_tier must be a tuple.")
        if len(self.pool_copies_by_tier) != self.max_tavern_tier:
            raise ValueError(
                "pool_copies_by_tier must contain one count for each Tavern tier."
            )
        for tier, copies in enumerate(self.pool_copies_by_tier, start=1):
            if isinstance(copies, bool) or not isinstance(copies, int):
                raise ValueError(
                    f"Pool copy count for Tavern tier {tier} must be an integer."
                )
            if copies < 1:
                raise ValueError(
                    f"Pool copy count for Tavern tier {tier} must be positive."
                )
        if self.board_limit < 1:
            raise ValueError("board_limit must be positive.")
        if self.hand_limit < 1:
            raise ValueError("hand_limit must be positive.")
        if self.combat_damage_cap < 1:
            raise ValueError("combat_damage_cap must be positive.")
        if self.rematch_avoidance_rounds < 0:
            raise ValueError("rematch_avoidance_rounds must not be negative.")
        if self.max_match_turns < 1:
            raise ValueError("max_match_turns must be positive.")

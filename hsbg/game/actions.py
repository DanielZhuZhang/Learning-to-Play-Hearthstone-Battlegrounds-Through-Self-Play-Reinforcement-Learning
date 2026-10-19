"""Explicit recruit actions that transform immutable player state."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import replace

from hsbg.game.config import GameRules
from hsbg.game.pool import MinionPool
from hsbg.game.shop import draw_shop_offers
from hsbg.game.state import MinionInstance, MinionZone, PlayerState


class InvalidActionError(ValueError):
    """Raised when a recruit action is not legal for the supplied state."""


def buy_offer(
    player: PlayerState,
    offer_id: str,
    instance_id: str,
    rules: GameRules,
) -> PlayerState:
    """Buy an existing shop offer into hand without executing its effects."""
    offer = next((entry for entry in player.shop if entry.offer_id == offer_id), None)
    if offer is None:
        raise InvalidActionError(f"Shop offer {offer_id!r} does not exist.")
    if player.gold < rules.buy_cost:
        raise InvalidActionError(
            f"Buying an offer costs {rules.buy_cost} gold; player has {player.gold}."
        )
    if len(player.hand) >= rules.hand_limit:
        raise InvalidActionError(f"Hand is full; limit is {rules.hand_limit} minions.")
    if any(minion.instance_id == instance_id for minion in player.minions):
        raise InvalidActionError(
            f"Minion instance ID {instance_id!r} already exists for this player."
        )

    purchased = MinionInstance(
        instance_id=instance_id,
        definition_id=offer.definition_id,
        zone=MinionZone.HAND,
        attack_bonus=offer.attack_bonus,
        health_bonus=offer.health_bonus,
        granted_keywords=offer.granted_keywords,
    )
    remaining_shop = tuple(entry for entry in player.shop if entry.offer_id != offer_id)
    return replace(
        player,
        gold=player.gold - rules.buy_cost,
        minions=player.minions + (purchased,),
        shop=remaining_shop,
    )


def play_minion(
    player: PlayerState,
    instance_id: str,
    rules: GameRules,
) -> PlayerState:
    """Move a minion from hand to the end of the board without resolving effects."""
    minion = next(
        (entry for entry in player.minions if entry.instance_id == instance_id),
        None,
    )
    if minion is None:
        raise InvalidActionError(f"Minion instance {instance_id!r} does not exist.")
    if minion.zone is not MinionZone.HAND:
        raise InvalidActionError(
            f"Minion instance {instance_id!r} is not in the player's hand."
        )
    if len(player.board) >= rules.board_limit:
        raise InvalidActionError(
            f"Board is full; limit is {rules.board_limit} minions."
        )

    played_minion = replace(minion, zone=MinionZone.BOARD)
    remaining_minions = tuple(
        entry for entry in player.minions if entry.instance_id != instance_id
    )
    return replace(player, minions=remaining_minions + (played_minion,))


def set_shop_frozen(player: PlayerState, frozen: bool) -> PlayerState:
    """Set whether the current shop should persist through the next refresh."""
    if not isinstance(frozen, bool):
        raise InvalidActionError("Shop frozen state must be a boolean.")
    return replace(player, shop_frozen=frozen)


def upgrade_tavern(player: PlayerState, rules: GameRules) -> PlayerState:
    """Pay the configured next-tier cost and raise the player's Tavern tier."""
    if player.tavern_tier >= rules.max_tavern_tier:
        raise InvalidActionError(
            f"Player is already at the maximum Tavern tier ({rules.max_tavern_tier})."
        )
    if player.tavern_tier < 1:
        raise InvalidActionError("Player Tavern tier must be at least 1.")

    cost = rules.tavern_upgrade_costs[player.tavern_tier - 1]
    if player.gold < cost:
        raise InvalidActionError(
            f"Upgrading to Tavern tier {player.tavern_tier + 1} costs {cost} gold; "
            f"player has {player.gold}."
        )
    return replace(
        player,
        gold=player.gold - cost,
        tavern_tier=player.tavern_tier + 1,
    )


def start_recruit_turn(
    player: PlayerState,
    pool: MinionPool,
    turn: int,
    eligible_card_ids: Sequence[str],
    offer_ids: Sequence[str],
    rng: random.Random,
    rules: GameRules,
) -> tuple[PlayerState, MinionPool]:
    """Set turn gold and retain a frozen shop or draw a fresh one."""
    if isinstance(turn, bool) or not isinstance(turn, int) or turn < 1:
        raise InvalidActionError("Recruit turn must be a positive integer.")

    gold = min(
        rules.starting_gold + (turn - 1) * rules.gold_per_turn,
        rules.max_gold,
    )
    if player.shop_frozen:
        return (
            replace(player, gold=gold, shop_frozen=False),
            pool,
        )

    _validate_shop_slots(player, offer_ids, rules)
    updated_pool = pool
    for offer in player.shop:
        updated_pool = updated_pool.return_copy(offer.definition_id)
    updated_pool, offers = draw_shop_offers(
        updated_pool, eligible_card_ids, offer_ids, rng
    )
    updated_player = replace(
        player,
        gold=gold,
        shop=offers,
        shop_frozen=False,
    )
    return updated_player, updated_pool


def sell_minion(
    player: PlayerState,
    pool: MinionPool,
    instance_id: str,
    rules: GameRules,
) -> tuple[PlayerState, MinionPool]:
    """Sell a board minion, returning its pool copy when it owns one."""
    minion = next(
        (entry for entry in player.minions if entry.instance_id == instance_id),
        None,
    )
    if minion is None:
        raise InvalidActionError(f"Minion instance {instance_id!r} does not exist.")
    if minion.zone is not MinionZone.BOARD:
        raise InvalidActionError(
            f"Minion instance {instance_id!r} is not on the player's board."
        )

    updated_pool = pool.return_copy(minion.definition_id) if minion.pool_owned else pool
    remaining_minions = tuple(
        entry for entry in player.minions if entry.instance_id != instance_id
    )
    updated_player = replace(
        player,
        gold=min(player.gold + rules.sell_refund, rules.max_gold),
        minions=remaining_minions,
    )
    return updated_player, updated_pool


def reorder_board(
    player: PlayerState,
    instance_ids: Sequence[str],
) -> PlayerState:
    """Reorder all current board minions while preserving hand order."""
    if isinstance(instance_ids, (str, bytes)) or not isinstance(instance_ids, Sequence):
        raise InvalidActionError("Board order must be a sequence of instance IDs.")
    if any(
        not isinstance(instance_id, str) or not instance_id.strip()
        for instance_id in instance_ids
    ):
        raise InvalidActionError("Board order must contain non-empty instance IDs.")

    board = player.board
    current_ids = {minion.instance_id for minion in board}
    requested_ids = tuple(instance_ids)
    if len(set(requested_ids)) != len(requested_ids):
        raise InvalidActionError("Board order must not contain duplicate instance IDs.")
    if set(requested_ids) != current_ids:
        raise InvalidActionError(
            "Board order must contain each current board minion exactly once."
        )

    minions_by_id = {minion.instance_id: minion for minion in board}
    reordered_board = tuple(minions_by_id[instance_id] for instance_id in requested_ids)
    board_positions = iter(reordered_board)
    updated_minions = tuple(
        next(board_positions) if minion.zone is MinionZone.BOARD else minion
        for minion in player.minions
    )
    return replace(player, minions=updated_minions)


def reroll_shop(
    player: PlayerState,
    pool: MinionPool,
    eligible_card_ids: Sequence[str],
    offer_ids: Sequence[str],
    rng: random.Random,
    rules: GameRules,
) -> tuple[PlayerState, MinionPool]:
    """Replace the player's shop, returning old offers before drawing new ones."""
    if player.gold < rules.reroll_cost:
        raise InvalidActionError(
            f"Rerolling the shop costs {rules.reroll_cost} gold; "
            f"player has {player.gold}."
        )
    _validate_shop_slots(player, offer_ids, rules)

    updated_pool = pool
    for offer in player.shop:
        updated_pool = updated_pool.return_copy(offer.definition_id)

    updated_pool, offers = draw_shop_offers(
        updated_pool, eligible_card_ids, offer_ids, rng
    )
    updated_player = replace(
        player,
        gold=player.gold - rules.reroll_cost,
        shop=offers,
        shop_frozen=False,
    )
    return updated_player, updated_pool


def _validate_shop_slots(
    player: PlayerState,
    offer_ids: Sequence[str],
    rules: GameRules,
) -> None:
    if isinstance(offer_ids, (str, bytes)) or not isinstance(offer_ids, Sequence):
        raise InvalidActionError("Shop offer IDs must be supplied as a sequence.")
    if player.tavern_tier > rules.max_tavern_tier:
        raise InvalidActionError(
            f"Player Tavern tier {player.tavern_tier} exceeds configured maximum "
            f"{rules.max_tavern_tier}."
        )
    expected_slots = rules.shop_slots_by_tier[player.tavern_tier - 1]
    if len(offer_ids) != expected_slots:
        raise InvalidActionError(
            f"Shop at Tavern tier {player.tavern_tier} requires {expected_slots} "
            f"offer IDs; received {len(offer_ids)}."
        )

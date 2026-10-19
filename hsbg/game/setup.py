"""Deterministic initial match setup from selected card definitions."""

from __future__ import annotations

import random
from collections.abc import Sequence

from hsbg.data.models import CardRecord
from hsbg.game.config import GameRules
from hsbg.game.pool import MinionPool, build_initial_pool
from hsbg.game.shop import draw_shop_offers
from hsbg.game.state import MatchPhase, MatchState, PlayerState


class MatchSetupError(ValueError):
    """Raised when initial match inputs do not describe a valid lobby."""


def create_initial_match(
    selected_cards: Sequence[CardRecord],
    rules: GameRules,
    player_ids: Sequence[str],
    offer_ids_by_player: Sequence[Sequence[str]],
    rng: random.Random,
    starting_health: int | None = None,
    starting_armor: int | None = None,
) -> tuple[MatchState, MinionPool]:
    """Create a turn-one recruit match and deal opening shops from one pool."""
    ids = _validate_player_ids(player_ids, rules.player_count)
    offer_id_groups = _validate_offer_id_groups(
        offer_ids_by_player,
        rules.player_count,
        rules.shop_slots_by_tier[rules.starting_tavern_tier - 1],
    )
    health = rules.starting_health if starting_health is None else starting_health
    armor = rules.starting_armor if starting_armor is None else starting_armor
    if isinstance(health, bool) or not isinstance(health, int) or health < 1:
        raise MatchSetupError("Starting health must be a positive integer.")
    if isinstance(armor, bool) or not isinstance(armor, int) or armor < 0:
        raise MatchSetupError("Starting armor must be a nonnegative integer.")
    if not isinstance(rng, random.Random):
        raise MatchSetupError("rng must be a random.Random instance.")

    pool = build_initial_pool(selected_cards, rules)
    eligible_card_ids = tuple(
        card.card_id
        for card in selected_cards
        if card.tier <= rules.starting_tavern_tier
    )
    players: list[PlayerState] = []
    for player_id, offer_ids in zip(ids, offer_id_groups, strict=True):
        pool, shop = draw_shop_offers(pool, eligible_card_ids, offer_ids, rng)
        players.append(
            PlayerState(
                player_id=player_id,
                health=health,
                armor=armor,
                gold=rules.starting_gold,
                tavern_tier=rules.starting_tavern_tier,
                shop=shop,
            )
        )

    return (
        MatchState(turn=1, phase=MatchPhase.RECRUIT, players=tuple(players)),
        pool,
    )


def _validate_player_ids(
    player_ids: Sequence[str], expected_count: int
) -> tuple[str, ...]:
    if isinstance(player_ids, (str, bytes)) or not isinstance(player_ids, Sequence):
        raise MatchSetupError("Player IDs must be a sequence of non-empty strings.")
    ids = tuple(player_ids)
    if any(
        not isinstance(player_id, str) or not player_id.strip() for player_id in ids
    ):
        raise MatchSetupError("Player IDs must contain only non-empty strings.")
    if len(ids) != expected_count:
        raise MatchSetupError(
            f"Expected {expected_count} player IDs; received {len(ids)}."
        )
    if len(set(ids)) != len(ids):
        raise MatchSetupError("Player IDs must be unique.")
    return ids


def _validate_offer_id_groups(
    offer_ids_by_player: Sequence[Sequence[str]],
    expected_count: int,
    expected_slots: int,
) -> tuple[tuple[str, ...], ...]:
    if isinstance(offer_ids_by_player, (str, bytes)) or not isinstance(
        offer_ids_by_player, Sequence
    ):
        raise MatchSetupError("Offer IDs must be supplied for each player.")
    groups = tuple(offer_ids_by_player)
    if len(groups) != expected_count:
        raise MatchSetupError(
            f"Expected offer IDs for {expected_count} players; received {len(groups)}."
        )
    for player_index, group in enumerate(groups):
        if isinstance(group, (str, bytes)) or not isinstance(group, Sequence):
            raise MatchSetupError(
                f"Offer IDs for player {player_index} must be a sequence."
            )
        if any(
            not isinstance(offer_id, str) or not offer_id.strip() for offer_id in group
        ):
            raise MatchSetupError(
                f"Offer IDs for player {player_index} must be non-empty strings."
            )
        if len(set(group)) != len(group):
            raise MatchSetupError(
                f"Offer IDs for player {player_index} must be unique."
            )
        if len(group) != expected_slots:
            raise MatchSetupError(
                f"Expected {expected_slots} opening offer IDs for player "
                f"{player_index}; received {len(group)}."
            )
    return tuple(tuple(group) for group in groups)

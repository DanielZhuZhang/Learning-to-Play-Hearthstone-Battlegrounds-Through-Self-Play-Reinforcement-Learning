"""Read and validate JSON iteration profiles (rules plus the enabled card pool)."""

from __future__ import annotations

from pathlib import Path

from hsbg.data.cards import load_card_library
from hsbg.data.json_fields import (
    JsonFieldError,
    object_fields,
    read_json,
    required_integer,
    required_integer_list,
    required_string,
    string_list,
)
from hsbg.data.models import CardEffectBinding, CardRecord, IterationProfile
from hsbg.game.config import GameRules


class ProfileDataError(ValueError):
    """Raised when a JSON profile does not match the supported schema."""


def load_profile(profile_path: str | Path) -> IterationProfile:
    """Load one profile and bind its enabled cards' effects from the card library."""
    profile, _cards = load_game_data(profile_path)
    return profile


def load_game_data(
    profile_path: str | Path,
) -> tuple[IterationProfile, tuple[CardRecord, ...]]:
    """Load a profile plus its enabled cards, in profile order, ready for ``GameEngine``.

    The profile's ``cards_file`` is resolved relative to the profile file. Every
    enabled card must exist in that library and be marked implemented.
    """
    path = Path(profile_path)
    try:
        if not path.is_file():
            raise JsonFieldError(f"Profile {path} does not exist.")
        fields = object_fields(
            read_json(path, "Profile"),
            context="profile",
            required={"iteration_id", "cards_file", "enabled_card_ids", "rules"},
            optional=set(),
        )
        iteration_id = required_string(fields, "iteration_id", "profile")
        cards_file = required_string(fields, "cards_file", "profile")
        enabled_card_ids = string_list(fields["enabled_card_ids"], "enabled_card_ids")
        rules = _parse_game_rules(fields["rules"])
    except JsonFieldError as exc:
        raise ProfileDataError(str(exc)) from exc

    library = {card.card_id: card for card in load_card_library(path.parent / cards_file)}
    missing = [card_id for card_id in enabled_card_ids if card_id not in library]
    if missing:
        raise ProfileDataError(
            f"Profile {path} enables card IDs not in {cards_file}: {', '.join(missing)}."
        )
    unimplemented = [
        card_id for card_id in enabled_card_ids if not library[card_id].implemented
    ]
    if unimplemented:
        raise ProfileDataError(
            f"Profile {path} enables cards that are not implemented: "
            f"{', '.join(unimplemented)}."
        )

    cards = tuple(library[card_id] for card_id in enabled_card_ids)
    try:
        profile = IterationProfile(
            iteration_id=iteration_id,
            cards_file=cards_file,
            enabled_card_ids=enabled_card_ids,
            card_effects=tuple(
                CardEffectBinding(card_id=card.card_id, effects=card.effects)
                for card in cards
            ),
            rules=rules,
        )
    except ValueError as exc:
        raise ProfileDataError(f"Profile {path} has invalid data: {exc}") from exc
    return profile, cards


def _parse_game_rules(value: object) -> GameRules:
    fields = object_fields(
        value,
        context="rules",
        required={
            "player_count",
            "starting_health",
            "starting_armor",
            "starting_gold",
            "gold_per_turn",
            "max_gold",
            "buy_cost",
            "sell_refund",
            "reroll_cost",
            "starting_tavern_tier",
            "max_tavern_tier",
            "tavern_upgrade_costs",
            "shop_slots_by_tier",
            "pool_copies_by_tier",
            "board_limit",
            "hand_limit",
            "combat_damage_cap",
            "rematch_avoidance_rounds",
            "max_match_turns",
        },
        optional=set(),
    )
    try:
        return GameRules(
            player_count=required_integer(fields, "player_count", "rules"),
            starting_health=required_integer(fields, "starting_health", "rules"),
            starting_armor=required_integer(fields, "starting_armor", "rules"),
            starting_gold=required_integer(fields, "starting_gold", "rules"),
            gold_per_turn=required_integer(fields, "gold_per_turn", "rules"),
            max_gold=required_integer(fields, "max_gold", "rules"),
            buy_cost=required_integer(fields, "buy_cost", "rules"),
            sell_refund=required_integer(fields, "sell_refund", "rules"),
            reroll_cost=required_integer(fields, "reroll_cost", "rules"),
            starting_tavern_tier=required_integer(
                fields, "starting_tavern_tier", "rules"
            ),
            max_tavern_tier=required_integer(fields, "max_tavern_tier", "rules"),
            tavern_upgrade_costs=required_integer_list(
                fields, "tavern_upgrade_costs", "rules"
            ),
            shop_slots_by_tier=required_integer_list(
                fields, "shop_slots_by_tier", "rules"
            ),
            pool_copies_by_tier=required_integer_list(
                fields, "pool_copies_by_tier", "rules"
            ),
            board_limit=required_integer(fields, "board_limit", "rules"),
            hand_limit=required_integer(fields, "hand_limit", "rules"),
            combat_damage_cap=required_integer(fields, "combat_damage_cap", "rules"),
            rematch_avoidance_rounds=required_integer(
                fields, "rematch_avoidance_rounds", "rules"
            ),
            max_match_turns=required_integer(fields, "max_match_turns", "rules"),
        )
    except JsonFieldError:
        raise
    except ValueError as exc:
        raise JsonFieldError(f"rules have invalid data: {exc}") from exc

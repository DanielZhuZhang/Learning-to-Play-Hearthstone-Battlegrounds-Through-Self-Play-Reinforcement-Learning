"""Read and validate JSON iteration profiles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypeVar

from hsbg.data.models import (
    BuffEffect,
    CardEffect,
    CardEffectBinding,
    CardOverride,
    DamageHeroEffect,
    EffectOperation,
    EffectTarget,
    EffectTrigger,
    IterationProfile,
    KeywordEffect,
    MinionKeyword,
    SummonEffect,
    TokenSpec,
)
from hsbg.game.config import GameRules


class ProfileDataError(ValueError):
    """Raised when a JSON profile does not match the supported schema."""


EnumValue = TypeVar("EnumValue", EffectTrigger, EffectTarget, MinionKeyword)


def load_profile(profile_path: str | Path) -> IterationProfile:
    """Load one profile, validating its JSON schema and model invariants."""
    path = Path(profile_path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ProfileDataError(
            f"Profile {path} contains invalid JSON at line {exc.lineno}, "
            f"column {exc.colno}: {exc.msg}."
        ) from exc
    except UnicodeDecodeError as exc:
        raise ProfileDataError(f"Profile {path} is not valid UTF-8.") from exc

    profile_data = _object_fields(
        data,
        context="profile",
        required={
            "iteration_id",
            "workbook",
            "sheet",
            "enabled_card_ids",
            "overrides",
            "card_effects",
            "rules",
        },
        optional=set(),
    )
    iteration_id = _required_string(profile_data, "iteration_id", "profile")
    workbook = _required_string(profile_data, "workbook", "profile")
    sheet = _required_string(profile_data, "sheet", "profile")
    enabled_card_ids = _string_list(
        profile_data["enabled_card_ids"], "enabled_card_ids"
    )
    overrides = _parse_overrides(profile_data["overrides"])
    card_effects = _parse_card_effects(profile_data["card_effects"], enabled_card_ids)
    rules = _parse_game_rules(profile_data["rules"])

    try:
        return IterationProfile(
            iteration_id=iteration_id,
            workbook=workbook,
            sheet=sheet,
            enabled_card_ids=enabled_card_ids,
            overrides=overrides,
            card_effects=card_effects,
            rules=rules,
        )
    except ValueError as exc:
        raise ProfileDataError(f"Profile {path} has invalid data: {exc}") from exc


def _parse_overrides(value: object) -> tuple[CardOverride, ...]:
    if not isinstance(value, list):
        raise ProfileDataError("'overrides' must be a list.")

    parsed: list[CardOverride] = []
    for index, item in enumerate(value):
        context = f"overrides[{index}]"
        fields = _object_fields(
            item,
            context=context,
            required={"card_id", "reason"},
            optional={"attack", "health", "tier", "text"},
        )
        card_id = _required_string(fields, "card_id", context)
        reason = _required_string(fields, "reason", context)
        try:
            parsed.append(
                CardOverride(
                    card_id=card_id,
                    reason=reason,
                    attack=_optional_integer(fields, "attack", context),
                    health=_optional_integer(fields, "health", context),
                    tier=_optional_integer(fields, "tier", context),
                    text=_optional_string(fields, "text", context),
                )
            )
        except ValueError as exc:
            raise ProfileDataError(f"{context} has invalid data: {exc}") from exc
    return tuple(parsed)


def _parse_card_effects(
    value: object, enabled_card_ids: tuple[str, ...]
) -> tuple[CardEffectBinding, ...]:
    if not isinstance(value, dict):
        raise ProfileDataError("'card_effects' must be an object keyed by card ID.")
    if any(not isinstance(card_id, str) for card_id in value):
        raise ProfileDataError("'card_effects' keys must be strings.")

    enabled_ids = set(enabled_card_ids)
    declared_ids = set(value)
    missing_ids = enabled_ids - declared_ids
    if missing_ids:
        missing = ", ".join(sorted(missing_ids))
        raise ProfileDataError(
            f"'card_effects' is missing enabled card IDs: {missing}."
        )
    unknown_ids = declared_ids - enabled_ids
    if unknown_ids:
        unknown = ", ".join(sorted(unknown_ids))
        raise ProfileDataError(
            f"'card_effects' has IDs not enabled in profile: {unknown}."
        )

    bindings: list[CardEffectBinding] = []
    for card_id in enabled_card_ids:
        raw_effects = value[card_id]
        if not isinstance(raw_effects, list):
            raise ProfileDataError(f"'card_effects' entry {card_id!r} must be a list.")
        effects = tuple(
            _parse_effect(effect, f"card_effects[{card_id!r}][{index}]")
            for index, effect in enumerate(raw_effects)
        )
        bindings.append(CardEffectBinding(card_id=card_id, effects=effects))
    return tuple(bindings)


def _parse_game_rules(value: object) -> GameRules:
    fields = _object_fields(
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
            player_count=_required_integer(fields, "player_count", "rules"),
            starting_health=_required_integer(fields, "starting_health", "rules"),
            starting_armor=_required_integer(fields, "starting_armor", "rules"),
            starting_gold=_required_integer(fields, "starting_gold", "rules"),
            gold_per_turn=_required_integer(fields, "gold_per_turn", "rules"),
            max_gold=_required_integer(fields, "max_gold", "rules"),
            buy_cost=_required_integer(fields, "buy_cost", "rules"),
            sell_refund=_required_integer(fields, "sell_refund", "rules"),
            reroll_cost=_required_integer(fields, "reroll_cost", "rules"),
            starting_tavern_tier=_required_integer(
                fields, "starting_tavern_tier", "rules"
            ),
            max_tavern_tier=_required_integer(fields, "max_tavern_tier", "rules"),
            tavern_upgrade_costs=_required_integer_list(
                fields, "tavern_upgrade_costs", "rules"
            ),
            shop_slots_by_tier=_required_integer_list(
                fields, "shop_slots_by_tier", "rules"
            ),
            pool_copies_by_tier=_required_integer_list(
                fields, "pool_copies_by_tier", "rules"
            ),
            board_limit=_required_integer(fields, "board_limit", "rules"),
            hand_limit=_required_integer(fields, "hand_limit", "rules"),
            combat_damage_cap=_required_integer(fields, "combat_damage_cap", "rules"),
            rematch_avoidance_rounds=_required_integer(
                fields, "rematch_avoidance_rounds", "rules"
            ),
            max_match_turns=_required_integer(fields, "max_match_turns", "rules"),
        )
    except ValueError as exc:
        raise ProfileDataError(f"rules have invalid data: {exc}") from exc


def _parse_effect(value: object, context: str) -> CardEffect:
    if not isinstance(value, dict):
        raise ProfileDataError(f"{context} must be a JSON object.")
    trigger_value = value.get("trigger")
    trigger = _enum_value(EffectTrigger, trigger_value, f"{context}.trigger")
    operation_value = value.get("operation")
    if not isinstance(operation_value, str):
        raise ProfileDataError(f"{context}.operation must be a string.")

    operation: EffectOperation
    if operation_value == "keyword":
        fields = _object_fields(
            value,
            context=context,
            required={"trigger", "operation", "target", "keyword"},
            optional=set(),
        )
        operation = KeywordEffect(
            target=_enum_value(EffectTarget, fields["target"], f"{context}.target"),
            keyword=_enum_value(MinionKeyword, fields["keyword"], f"{context}.keyword"),
        )
    elif operation_value == "summon":
        fields = _object_fields(
            value,
            context=context,
            required={"trigger", "operation", "token", "count"},
            optional=set(),
        )
        operation = SummonEffect(
            token=_parse_token(fields["token"], f"{context}.token"),
            count=_required_integer(fields, "count", context),
        )
    elif operation_value == "buff":
        fields = _object_fields(
            value,
            context=context,
            required={"trigger", "operation", "target", "attack", "health"},
            optional={"keywords"},
        )
        keywords = _keyword_list(fields.get("keywords", []), f"{context}.keywords")
        operation = BuffEffect(
            target=_enum_value(EffectTarget, fields["target"], f"{context}.target"),
            attack=_required_integer(fields, "attack", context),
            health=_required_integer(fields, "health", context),
            keywords=keywords,
        )
    elif operation_value == "damage_hero":
        fields = _object_fields(
            value,
            context=context,
            required={"trigger", "operation", "amount"},
            optional=set(),
        )
        operation = DamageHeroEffect(
            amount=_required_integer(fields, "amount", context)
        )
    else:
        raise ProfileDataError(
            f"{context}.operation has unsupported value {operation_value!r}."
        )

    try:
        return CardEffect(trigger=trigger, operation=operation)
    except ValueError as exc:
        raise ProfileDataError(f"{context} has invalid data: {exc}") from exc


def _parse_token(value: object, context: str) -> TokenSpec:
    fields = _object_fields(
        value,
        context=context,
        required={"name", "attack", "health"},
        optional={"keywords"},
    )
    keywords = _keyword_list(fields.get("keywords", []), f"{context}.keywords")
    try:
        return TokenSpec(
            name=_required_string(fields, "name", context),
            attack=_required_integer(fields, "attack", context),
            health=_required_integer(fields, "health", context),
            keywords=keywords,
        )
    except ValueError as exc:
        raise ProfileDataError(f"{context} has invalid data: {exc}") from exc


def _keyword_list(value: object, context: str) -> tuple[MinionKeyword, ...]:
    if not isinstance(value, list):
        raise ProfileDataError(f"{context} must be a list of keywords.")
    return tuple(
        _enum_value(MinionKeyword, item, f"{context}[{index}]")
        for index, item in enumerate(value)
    )


def _enum_value(enum_type: type[EnumValue], value: object, context: str) -> EnumValue:
    if not isinstance(value, str):
        raise ProfileDataError(f"{context} must be a string.")
    try:
        return enum_type(value)
    except ValueError as exc:
        allowed = ", ".join(member.value for member in enum_type)
        raise ProfileDataError(
            f"{context} has unsupported value {value!r}; expected one of: {allowed}."
        ) from exc


def _required_integer(fields: dict[str, object], name: str, context: str) -> int:
    value = fields[name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProfileDataError(f"{context}.{name} must be an integer.")
    return value


def _required_integer_list(
    fields: dict[str, object], name: str, context: str
) -> tuple[int, ...]:
    value = fields[name]
    if not isinstance(value, list):
        raise ProfileDataError(f"{context}.{name} must be a list of integers.")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise ProfileDataError(f"{context}.{name} must contain only integers.")
    return tuple(value)


def _object_fields(
    value: object,
    *,
    context: str,
    required: set[str],
    optional: set[str],
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ProfileDataError(f"{context} must be a JSON object.")
    if any(not isinstance(key, str) for key in value):
        raise ProfileDataError(f"{context} keys must be strings.")

    missing = required - value.keys()
    if missing:
        raise ProfileDataError(
            f"{context} is missing required fields: {', '.join(sorted(missing))}."
        )
    unknown = value.keys() - required - optional
    if unknown:
        raise ProfileDataError(
            f"{context} has unknown fields: {', '.join(sorted(unknown))}."
        )
    return {key: value[key] for key in value}


def _required_string(fields: dict[str, object], name: str, context: str) -> str:
    value = fields[name]
    if not isinstance(value, str):
        raise ProfileDataError(f"{context}.{name} must be a string.")
    return value


def _string_list(value: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ProfileDataError(f"{field_name!r} must be a list of strings.")
    if any(not isinstance(item, str) for item in value):
        raise ProfileDataError(f"{field_name!r} must contain only strings.")
    return tuple(value)


def _optional_integer(fields: dict[str, object], name: str, context: str) -> int | None:
    value = fields.get(name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProfileDataError(f"{context}.{name} must be an integer or null.")
    return value


def _optional_string(fields: dict[str, object], name: str, context: str) -> str | None:
    value = fields.get(name)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ProfileDataError(f"{context}.{name} must be a string or null.")
    return value

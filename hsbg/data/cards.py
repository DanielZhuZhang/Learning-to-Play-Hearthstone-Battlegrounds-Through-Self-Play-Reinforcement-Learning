"""Read and validate the JSON card library (stats, text, and effects)."""

from __future__ import annotations

from pathlib import Path

from hsbg.data.json_fields import (
    JsonFieldError,
    enum_value,
    keyword_list,
    object_fields,
    optional_string,
    read_json,
    required_bool,
    required_integer,
    required_string,
)
from hsbg.data.models import (
    BuffEffect,
    CardEffect,
    CardRecord,
    DamageHeroEffect,
    EffectOperation,
    EffectTarget,
    EffectTrigger,
    KeywordEffect,
    MinionKeyword,
    SummonEffect,
    TokenSpec,
)


class CardDataError(ValueError):
    """Raised when the card library does not match the supported schema."""


CARD_FIELDS = {"id", "name", "tier", "tribe", "type", "attack", "health", "text", "implemented"}


def load_card_library(cards_path: str | Path) -> tuple[CardRecord, ...]:
    """Load every card in file order, validating fields and effect declarations."""
    path = Path(cards_path)
    try:
        if not path.is_file():
            raise JsonFieldError(f"Cards file {path} does not exist.")
        data = object_fields(
            read_json(path, "Cards file"),
            context="cards file",
            required={"cards"},
            optional=set(),
        )
        if not isinstance(data["cards"], list):
            raise JsonFieldError("'cards' must be a list of card objects.")
        cards: list[CardRecord] = []
        seen: set[str] = set()
        for index, value in enumerate(data["cards"]):
            card = _parse_card(value, index)
            if card.card_id in seen:
                raise JsonFieldError(f"cards[{index}] repeats card ID {card.card_id!r}.")
            seen.add(card.card_id)
            cards.append(card)
        return tuple(cards)
    except ValueError as exc:
        # Field errors and model invariant errors (e.g. a buff that changes nothing).
        raise CardDataError(f"Cards file {path}: {exc}") from exc


def _parse_card(value: object, index: int) -> CardRecord:
    context = f"cards[{index}]"
    if isinstance(value, dict) and isinstance(value.get("id"), str):
        context = f"card {value['id']!r}"
    fields = object_fields(
        value, context=context, required=CARD_FIELDS, optional={"effects"}
    )
    implemented = required_bool(fields, "implemented", context)
    if implemented and "effects" not in fields:
        raise JsonFieldError(
            f"{context} is implemented, so it must list its effects (use [] for none)."
        )
    if not implemented and "effects" in fields:
        raise JsonFieldError(f"{context} is not implemented, so it must not list effects.")
    raw_effects = fields.get("effects", [])
    if not isinstance(raw_effects, list):
        raise JsonFieldError(f"{context}.effects must be a list.")
    try:
        effects = tuple(
            parse_effect(effect, f"{context}.effects[{effect_index}]")
            for effect_index, effect in enumerate(raw_effects)
        )
        return CardRecord(
            card_id=required_string(fields, "id", context),
            name=required_string(fields, "name", context),
            tier=required_integer(fields, "tier", context),
            tribe=optional_string(fields, "tribe", context),
            card_type=optional_string(fields, "type", context),
            attack=required_integer(fields, "attack", context),
            health=required_integer(fields, "health", context),
            text=required_string(fields, "text", context),
            implemented=implemented,
            effects=effects,
        )
    except ValueError as exc:
        if isinstance(exc, JsonFieldError):
            raise
        raise JsonFieldError(f"{context} has invalid data: {exc}") from exc


def parse_effect(value: object, context: str) -> CardEffect:
    if not isinstance(value, dict):
        raise JsonFieldError(f"{context} must be a JSON object.")
    trigger = enum_value(EffectTrigger, value.get("trigger"), f"{context}.trigger")
    operation_value = value.get("operation")
    if not isinstance(operation_value, str):
        raise JsonFieldError(f"{context}.operation must be a string.")

    try:
        operation = _parse_operation(value, operation_value, context)
        return CardEffect(trigger=trigger, operation=operation)
    except JsonFieldError:
        raise
    except ValueError as exc:
        raise JsonFieldError(f"{context} has invalid data: {exc}") from exc


def _parse_operation(
    value: dict[object, object], operation_value: str, context: str
) -> EffectOperation:
    operation: EffectOperation
    if operation_value == "keyword":
        fields = object_fields(
            value,
            context=context,
            required={"trigger", "operation", "target", "keyword"},
            optional=set(),
        )
        operation = KeywordEffect(
            target=enum_value(EffectTarget, fields["target"], f"{context}.target"),
            keyword=enum_value(MinionKeyword, fields["keyword"], f"{context}.keyword"),
        )
    elif operation_value == "summon":
        fields = object_fields(
            value,
            context=context,
            required={"trigger", "operation", "token", "count"},
            optional=set(),
        )
        operation = SummonEffect(
            token=_parse_token(fields["token"], f"{context}.token"),
            count=required_integer(fields, "count", context),
        )
    elif operation_value == "buff":
        fields = object_fields(
            value,
            context=context,
            required={"trigger", "operation", "target", "attack", "health"},
            optional={"keywords"},
        )
        operation = BuffEffect(
            target=enum_value(EffectTarget, fields["target"], f"{context}.target"),
            attack=required_integer(fields, "attack", context),
            health=required_integer(fields, "health", context),
            keywords=keyword_list(fields.get("keywords", []), f"{context}.keywords"),
        )
    elif operation_value == "damage_hero":
        fields = object_fields(
            value,
            context=context,
            required={"trigger", "operation", "amount"},
            optional=set(),
        )
        operation = DamageHeroEffect(amount=required_integer(fields, "amount", context))
    else:
        raise JsonFieldError(
            f"{context}.operation has unsupported value {operation_value!r}."
        )
    return operation


def _parse_token(value: object, context: str) -> TokenSpec:
    fields = object_fields(
        value,
        context=context,
        required={"name", "attack", "health"},
        optional={"keywords"},
    )
    try:
        return TokenSpec(
            name=required_string(fields, "name", context),
            attack=required_integer(fields, "attack", context),
            health=required_integer(fields, "health", context),
            keywords=keyword_list(fields.get("keywords", []), f"{context}.keywords"),
        )
    except ValueError as exc:
        if isinstance(exc, JsonFieldError):
            raise
        raise JsonFieldError(f"{context} has invalid data: {exc}") from exc

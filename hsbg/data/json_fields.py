"""Strict field readers shared by the card-library and profile JSON loaders."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypeVar

from hsbg.data.models import EffectTarget, EffectTrigger, MinionKeyword


class JsonFieldError(ValueError):
    """Raised when a JSON value does not match the expected schema."""


EnumValue = TypeVar("EnumValue", EffectTrigger, EffectTarget, MinionKeyword)


def object_fields(
    value: object,
    *,
    context: str,
    required: set[str],
    optional: set[str],
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise JsonFieldError(f"{context} must be a JSON object.")
    if any(not isinstance(key, str) for key in value):
        raise JsonFieldError(f"{context} keys must be strings.")

    missing = required - value.keys()
    if missing:
        raise JsonFieldError(
            f"{context} is missing required fields: {', '.join(sorted(missing))}."
        )
    unknown = value.keys() - required - optional
    if unknown:
        raise JsonFieldError(
            f"{context} has unknown fields: {', '.join(sorted(unknown))}."
        )
    return {key: value[key] for key in value}


def enum_value(enum_type: type[EnumValue], value: object, context: str) -> EnumValue:
    if not isinstance(value, str):
        raise JsonFieldError(f"{context} must be a string.")
    try:
        return enum_type(value)
    except ValueError as exc:
        allowed = ", ".join(member.value for member in enum_type)
        raise JsonFieldError(
            f"{context} has unsupported value {value!r}; expected one of: {allowed}."
        ) from exc


def keyword_list(value: object, context: str) -> tuple[MinionKeyword, ...]:
    if not isinstance(value, list):
        raise JsonFieldError(f"{context} must be a list of keywords.")
    return tuple(
        enum_value(MinionKeyword, item, f"{context}[{index}]")
        for index, item in enumerate(value)
    )


def required_integer(fields: dict[str, object], name: str, context: str) -> int:
    value = fields[name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise JsonFieldError(f"{context}.{name} must be an integer.")
    return value


def required_integer_list(
    fields: dict[str, object], name: str, context: str
) -> tuple[int, ...]:
    value = fields[name]
    if not isinstance(value, list):
        raise JsonFieldError(f"{context}.{name} must be a list of integers.")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise JsonFieldError(f"{context}.{name} must contain only integers.")
    return tuple(value)


def required_string(fields: dict[str, object], name: str, context: str) -> str:
    value = fields[name]
    if not isinstance(value, str):
        raise JsonFieldError(f"{context}.{name} must be a string.")
    return value


def optional_string(fields: dict[str, object], name: str, context: str) -> str | None:
    value = fields.get(name)
    if value is None:
        return None
    if not isinstance(value, str):
        raise JsonFieldError(f"{context}.{name} must be a string or null.")
    return value


def required_bool(fields: dict[str, object], name: str, context: str) -> bool:
    value = fields[name]
    if not isinstance(value, bool):
        raise JsonFieldError(f"{context}.{name} must be true or false.")
    return value


def string_list(value: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise JsonFieldError(f"{field_name!r} must be a list of strings.")
    if any(not isinstance(item, str) for item in value):
        raise JsonFieldError(f"{field_name!r} must contain only strings.")
    return tuple(value)


def read_json(path: Path, label: str) -> object:
    """Read a UTF-8 JSON file, reporting decode errors with their location."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise JsonFieldError(
            f"{label} {path} contains invalid JSON at line {exc.lineno}, "
            f"column {exc.colno}: {exc.msg}."
        ) from exc
    except UnicodeDecodeError as exc:
        raise JsonFieldError(f"{label} {path} is not valid UTF-8.") from exc

import json
from pathlib import Path

import pytest

from hsbg.data.models import (
    CardEffect,
    CardEffectBinding,
    CardOverride,
    EffectTrigger,
    IterationProfile,
    TokenSpec,
    SummonEffect,
)
from hsbg.data.profiles import ProfileDataError, load_profile
from hsbg.data.selection import select_enabled_cards
from hsbg.data.workbook import load_sheet
from hsbg.game.config import GameRules

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_profile(path: Path, profile: object) -> None:
    path.write_text(json.dumps(profile), encoding="utf-8")


def _profile_data() -> dict[str, object]:
    return {
        "iteration_id": "test-15.6",
        "workbook": "season -1 (beta).xlsx",
        "sheet": "15.6",
        "rules": {
            "player_count": 8,
            "starting_health": 30,
            "starting_armor": 0,
            "starting_gold": 3,
            "gold_per_turn": 1,
            "max_gold": 10,
            "buy_cost": 3,
            "sell_refund": 1,
            "reroll_cost": 1,
            "starting_tavern_tier": 1,
            "max_tavern_tier": 6,
            "tavern_upgrade_costs": [5, 7, 8, 9, 10],
            "shop_slots_by_tier": [3, 4, 5, 6, 7, 7],
            "pool_copies_by_tier": [16, 15, 13, 11, 9, 7],
            "board_limit": 7,
            "hand_limit": 10,
            "combat_damage_cap": 15,
            "rematch_avoidance_rounds": 1,
            "max_match_turns": 100,
        },
        "enabled_card_ids": ["15.6:1"],
        "overrides": [
            {
                "card_id": "15.6:1",
                "reason": "Testing a balance change.",
                "attack": 2,
                "text": "Updated text.",
            }
        ],
        "card_effects": {
            "15.6:1": [
                {
                    "trigger": "battlecry",
                    "operation": "summon",
                    "count": 1,
                    "token": {
                        "name": "Cat",
                        "attack": 1,
                        "health": 1,
                    },
                }
            ]
        },
    }


def test_load_profile_parses_profile_and_explicit_override(tmp_path: Path) -> None:
    path = tmp_path / "profile.json"
    _write_profile(path, _profile_data())

    profile = load_profile(path)

    assert isinstance(profile, IterationProfile)
    assert profile.iteration_id == "test-15.6"
    assert profile.workbook == "season -1 (beta).xlsx"
    assert profile.sheet == "15.6"
    assert profile.rules == GameRules()
    assert profile.enabled_card_ids == ("15.6:1",)
    assert profile.overrides == (
        CardOverride(
            card_id="15.6:1",
            reason="Testing a balance change.",
            attack=2,
            text="Updated text.",
        ),
    )
    assert profile.card_effects == (
        CardEffectBinding(
            card_id="15.6:1",
            effects=(
                CardEffect(
                    trigger=EffectTrigger.BATTLECRY,
                    operation=SummonEffect(
                        token=TokenSpec(name="Cat", attack=1, health=1),
                        count=1,
                    ),
                ),
            ),
        ),
    )
    assert isinstance(profile.card_effects[0].effects[0].operation, SummonEffect)


def test_load_profile_accepts_empty_selections_and_overrides(tmp_path: Path) -> None:
    path = tmp_path / "empty-profile.json"
    data = _profile_data()
    data["enabled_card_ids"] = []
    data["overrides"] = []
    data["card_effects"] = {}
    _write_profile(path, data)

    profile = load_profile(path)

    assert profile.enabled_card_ids == ()
    assert profile.overrides == ()
    assert profile.card_effects == ()


def test_missing_profile_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_profile(tmp_path / "missing.json")


def test_invalid_json_reports_path_and_location(tmp_path: Path) -> None:
    path = tmp_path / "invalid.json"
    path.write_text('{"iteration_id":', encoding="utf-8")

    with pytest.raises(ProfileDataError, match=r"invalid JSON at line 1, column"):
        load_profile(path)


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ([], "profile must be a JSON object"),
        ({**_profile_data(), "extra": True}, "unknown fields: extra"),
        (
            {key: value for key, value in _profile_data().items() if key != "sheet"},
            "missing required fields: sheet",
        ),
        ({**_profile_data(), "enabled_card_ids": "15.6:1"}, "must be a list"),
        ({**_profile_data(), "enabled_card_ids": [1]}, "only strings"),
        (
            {**_profile_data(), "enabled_card_ids": ["15.6:1", "15.6:1"]},
            "must not contain duplicates",
        ),
    ],
)
def test_invalid_profile_schema_and_model_data_are_reported(
    tmp_path: Path, data: object, message: str
) -> None:
    path = tmp_path / "invalid-profile.json"
    _write_profile(path, data)

    with pytest.raises(ProfileDataError, match=message):
        load_profile(path)


@pytest.mark.parametrize(
    ("changes", "remove_reason", "message"),
    [
        ({"reason": None}, False, r"overrides\[0\]\.reason must be a string"),
        (
            {"unknown": "field"},
            False,
            r"overrides\[0\] has unknown fields: unknown",
        ),
        ({"attack": True}, False, r"overrides\[0\]\.attack must be an integer"),
        ({"attack": 1.5}, False, r"overrides\[0\]\.attack must be an integer"),
        (
            {"attack": -1},
            False,
            r"overrides\[0\] has invalid data: Override attack",
        ),
        ({}, True, r"overrides\[0\] is missing required fields: reason"),
    ],
)
def test_invalid_override_schema_and_values_are_reported(
    tmp_path: Path,
    changes: dict[str, object],
    remove_reason: bool,
    message: str,
) -> None:
    path = tmp_path / "invalid-override.json"
    data = _profile_data()
    override = dict(data["overrides"][0])  # type: ignore[index]
    if remove_reason:
        del override["reason"]
    override.update(changes)
    data["overrides"] = [override]
    _write_profile(path, data)

    with pytest.raises(ProfileDataError, match=message):
        load_profile(path)


def test_override_for_disabled_card_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "disabled-override.json"
    data = _profile_data()
    data["enabled_card_ids"] = []
    data["card_effects"] = {}
    _write_profile(path, data)

    with pytest.raises(ProfileDataError, match="Overrides must target enabled cards"):
        load_profile(path)


def test_initial_profile_declares_effects_for_every_enabled_card() -> None:
    profile = load_profile(PROJECT_ROOT / "data" / "profiles" / "initial-15.6.json")
    selected_cards = select_enabled_cards(
        profile,
        load_sheet(PROJECT_ROOT / profile.workbook, profile.sheet),
    )

    assert len(profile.enabled_card_ids) == 29
    assert profile.rules == GameRules()
    assert tuple(binding.card_id for binding in profile.card_effects) == (
        profile.enabled_card_ids
    )
    assert tuple(card.card_id for card in selected_cards) == profile.enabled_card_ids
    assert (
        next(
            binding for binding in profile.card_effects if binding.card_id == "15.6:27"
        ).effects
        == ()
    )
    amalgam = next(card for card in selected_cards if card.card_id == "15.6:27")
    assert amalgam.card_type == "All"


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        (
            {
                "card_effects": {
                    "15.6:1": [{"trigger": "start_of_turn", "operation": "buff"}]
                }
            },
            r"trigger has unsupported value 'start_of_turn'",
        ),
        (
            {
                "card_effects": {
                    "15.6:1": [{"trigger": "battlecry", "operation": "discover"}]
                }
            },
            r"operation has unsupported value 'discover'",
        ),
        (
            {"card_effects": {}},
            r"'card_effects' is missing enabled card IDs: 15.6:1",
        ),
        (
            {"card_effects": {"15.6:1": [], "15.6:2": []}},
            r"'card_effects' has IDs not enabled in profile: 15.6:2",
        ),
        (
            {
                "card_effects": {
                    "15.6:1": [
                        {
                            "trigger": "battlecry",
                            "operation": "summon",
                            "count": True,
                            "token": {"name": "Cat", "attack": 1, "health": 1},
                        }
                    ]
                }
            },
            r"count must be an integer",
        ),
    ],
)
def test_invalid_card_effect_declarations_are_reported(
    tmp_path: Path, changes: dict[str, object], message: str
) -> None:
    path = tmp_path / "invalid-effects.json"
    data = _profile_data()
    data.update(changes)
    _write_profile(path, data)

    with pytest.raises(ProfileDataError, match=message):
        load_profile(path)


@pytest.mark.parametrize(
    ("rules", "message"),
    [
        ({}, r"rules is missing required fields:"),
        (
            {
                **_profile_data()["rules"],  # type: ignore[misc]
                "max_gold": 2,
            },
            "max_gold must be at least starting_gold",
        ),
        (
            {
                **_profile_data()["rules"],  # type: ignore[misc]
                "buy_cost": True,
            },
            r"rules\.buy_cost must be an integer",
        ),
        (
            {
                **_profile_data()["rules"],  # type: ignore[misc]
                "extra": 1,
            },
            "rules has unknown fields: extra",
        ),
        (
            {
                **_profile_data()["rules"],  # type: ignore[misc]
                "pool_copies_by_tier": [16, 15, True, 11, 9, 7],
            },
            r"rules\.pool_copies_by_tier must contain only integers",
        ),
        (
            {
                **_profile_data()["rules"],  # type: ignore[misc]
                "tavern_upgrade_costs": [5, 7, 8, True, 10],
            },
            r"rules\.tavern_upgrade_costs must contain only integers",
        ),
    ],
)
def test_invalid_rules_are_reported(
    tmp_path: Path, rules: object, message: str
) -> None:
    path = tmp_path / "invalid-rules.json"
    data = _profile_data()
    data["rules"] = rules
    _write_profile(path, data)

    with pytest.raises(ProfileDataError, match=message):
        load_profile(path)

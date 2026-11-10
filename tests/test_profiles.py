import json
from pathlib import Path

import pytest

from hsbg.data.cards import CardDataError
from hsbg.data.profiles import ProfileDataError, load_game_data, load_profile
from hsbg.game.config import GameRules

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INITIAL_PROFILE = PROJECT_ROOT / "data" / "profiles" / "initial-15.6.json"

_RULES = {
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
}

_CARDS = [
    {
        "id": "t:1",
        "name": "Taunter",
        "tier": 1,
        "tribe": None,
        "type": None,
        "attack": 1,
        "health": 3,
        "text": "Taunt",
        "implemented": True,
        "effects": [
            {"trigger": "passive", "operation": "keyword", "target": "self", "keyword": "taunt"}
        ],
    },
    {
        "id": "t:2",
        "name": "Plain",
        "tier": 2,
        "tribe": "Beast",
        "type": "Beast",
        "attack": 2,
        "health": 2,
        "text": "",
        "implemented": True,
        "effects": [],
    },
    {
        "id": "t:3",
        "name": "Aura",
        "tier": 1,
        "tribe": None,
        "type": None,
        "attack": 2,
        "health": 2,
        "text": "Adjacent minions have +1 Attack.",
        "implemented": False,
    },
]


def _write_game(tmp_path: Path, **changes: object) -> Path:
    """Write ``data/cards.json`` and ``data/profiles/test.json`` under ``tmp_path``."""
    (tmp_path / "profiles").mkdir()
    (tmp_path / "cards.json").write_text(json.dumps({"cards": _CARDS}), encoding="utf-8")
    profile: dict[str, object] = {
        "iteration_id": "test",
        "cards_file": "../cards.json",
        "rules": _RULES,
        "enabled_card_ids": ["t:2", "t:1"],
    }
    profile.update(changes)
    path = tmp_path / "profiles" / "test.json"
    path.write_text(json.dumps(profile), encoding="utf-8")
    return path


def test_profile_resolves_cards_file_and_keeps_profile_order(tmp_path: Path) -> None:
    path = _write_game(tmp_path)
    profile, cards = load_game_data(path)

    assert load_profile(path) == profile
    assert profile.iteration_id == "test"
    assert profile.rules == GameRules()
    assert profile.enabled_card_ids == ("t:2", "t:1")
    assert tuple(card.card_id for card in cards) == ("t:2", "t:1")
    assert tuple(binding.card_id for binding in profile.card_effects) == ("t:2", "t:1")
    assert profile.card_effects[1].effects == cards[1].effects


def test_initial_profile_enables_29_implemented_cards() -> None:
    profile, cards = load_game_data(INITIAL_PROFILE)

    assert len(cards) == 29
    assert profile.rules == GameRules()
    assert tuple(card.card_id for card in cards) == profile.enabled_card_ids
    assert all(card.implemented for card in cards)
    amalgam = next(card for card in cards if card.card_id == "15.6:27")
    assert amalgam.card_type == "All"
    assert amalgam.effects == ()


@pytest.mark.parametrize(
    ("changes", "error", "message"),
    [
        ({"enabled_card_ids": ["t:9"]}, ProfileDataError, r"not in \.\./cards\.json: t:9"),
        ({"enabled_card_ids": ["t:3"]}, ProfileDataError, r"not implemented: t:3"),
        ({"enabled_card_ids": ["t:1", "t:1"]}, ProfileDataError, r"duplicates"),
        ({"cards_file": "missing.json"}, CardDataError, r"does not exist"),
        ({"workbook": "x.xlsx"}, ProfileDataError, r"profile has unknown fields: workbook"),
        ({"enabled_card_ids": "t:1"}, ProfileDataError, r"must be a list of strings"),
    ],
)
def test_invalid_profiles_are_reported(
    tmp_path: Path, changes: dict[str, object], error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        load_profile(_write_game(tmp_path, **changes))


def test_missing_and_malformed_profiles_are_reported(tmp_path: Path) -> None:
    with pytest.raises(ProfileDataError, match="does not exist"):
        load_profile(tmp_path / "missing.json")

    broken = tmp_path / "broken.json"
    broken.write_text('{"iteration_id": ', encoding="utf-8")
    with pytest.raises(ProfileDataError, match="invalid JSON at line 1"):
        load_profile(broken)


@pytest.mark.parametrize(
    ("rules", "message"),
    [
        ({}, r"rules is missing required fields:"),
        ({**_RULES, "max_gold": 2}, "max_gold must be at least starting_gold"),
        ({**_RULES, "buy_cost": True}, r"rules\.buy_cost must be an integer"),
        ({**_RULES, "extra": 1}, "rules has unknown fields: extra"),
        (
            {**_RULES, "pool_copies_by_tier": [16, 15, True, 11, 9, 7]},
            r"rules\.pool_copies_by_tier must contain only integers",
        ),
        (
            {**_RULES, "tavern_upgrade_costs": [5, 7, 8, True, 10]},
            r"rules\.tavern_upgrade_costs must contain only integers",
        ),
    ],
)
def test_invalid_rules_are_reported(tmp_path: Path, rules: object, message: str) -> None:
    with pytest.raises(ProfileDataError, match=message):
        load_profile(_write_game(tmp_path, rules=rules))

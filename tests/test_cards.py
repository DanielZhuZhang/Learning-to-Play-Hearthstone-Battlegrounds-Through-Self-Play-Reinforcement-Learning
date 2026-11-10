import json
from pathlib import Path

import pytest

from hsbg.data.cards import CardDataError, load_card_library
from hsbg.data.models import (
    BuffEffect,
    EffectTarget,
    EffectTrigger,
    KeywordEffect,
    MinionKeyword,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CARDS_PATH = PROJECT_ROOT / "data" / "cards.json"


def _card(card_id: str = "t:1", **changes: object) -> dict[str, object]:
    card: dict[str, object] = {
        "id": card_id,
        "name": "Test Minion",
        "tier": 1,
        "tribe": None,
        "type": None,
        "attack": 1,
        "health": 1,
        "text": "",
        "implemented": True,
        "effects": [],
    }
    card.update(changes)
    return card


def _write(tmp_path: Path, *cards: dict[str, object]) -> Path:
    path = tmp_path / "cards.json"
    path.write_text(json.dumps({"cards": list(cards)}), encoding="utf-8")
    return path


def test_library_holds_every_sheet_card_with_unique_ids() -> None:
    cards = load_card_library(CARDS_PATH)

    assert len(cards) == 1092
    assert len({card.card_id for card in cards}) == len(cards)
    assert {card.card_id.split(":")[0] for card in cards} >= {"15.6", "16.0", "17.2"}


def test_library_keeps_stats_text_and_effects_together() -> None:
    cards = {card.card_id: card for card in load_card_library(CARDS_PATH)}

    overseer = cards["15.6:17"]
    assert (overseer.name, overseer.tier, overseer.tribe) == ("Nathrezim Overseer", 2, "Demon")
    assert (overseer.attack, overseer.health) == (2, 4)
    assert overseer.text == "Battlecry: Give a friendly Demon +2/+2."
    assert overseer.implemented
    (effect,) = overseer.effects
    assert effect.trigger is EffectTrigger.BATTLECRY
    assert effect.operation == BuffEffect(EffectTarget.FRIENDLY_DEMON, 2, 2)

    dire_wolf = cards["15.6:2"]
    assert not dire_wolf.implemented and dire_wolf.effects == ()
    assert cards["15.6:27"].card_type == "All"


def test_keyword_only_cards_carry_their_passive_keywords() -> None:
    cards = load_card_library(CARDS_PATH)
    keyword_only = [
        card
        for card in cards
        if card.text.strip()
        and set(card.text.replace("Divine Shield", "").split()) <= {"Taunt", "Poisonous"}
    ]

    assert keyword_only
    for card in keyword_only:
        assert card.implemented, card.card_id
        keywords = {
            effect.operation.keyword
            for effect in card.effects
            if isinstance(effect.operation, KeywordEffect)
        }
        expected = {
            keyword
            for keyword, word in (
                (MinionKeyword.TAUNT, "Taunt"),
                (MinionKeyword.DIVINE_SHIELD, "Divine Shield"),
                (MinionKeyword.POISONOUS, "Poisonous"),
            )
            if word in card.text
        }
        assert keywords == expected, card.card_id


@pytest.mark.parametrize(
    ("cards", "message"),
    [
        ((_card(effects=None),), r"effects must be a list"),
        ((_card(implemented=False),), r"not implemented, so it must not list effects"),
        (({k: v for k, v in _card().items() if k != "effects"},), r"must list its effects"),
        ((_card(), _card()), r"repeats card ID 't:1'"),
        ((_card(tier=0),), r"card 't:1' has invalid data: Card tier must be positive"),
        ((_card(attack="2"),), r"card 't:1'\.attack must be an integer"),
        ((_card(extra=1),), r"card 't:1' has unknown fields: extra"),
        (
            (_card(effects=[{"trigger": "start_of_turn", "operation": "buff"}]),),
            r"trigger has unsupported value 'start_of_turn'",
        ),
        (
            (_card(effects=[{"trigger": "battlecry", "operation": "discover"}]),),
            r"operation has unsupported value 'discover'",
        ),
        (
            (
                _card(
                    effects=[
                        {
                            "trigger": "battlecry",
                            "operation": "summon",
                            "count": True,
                            "token": {"name": "Cat", "attack": 1, "health": 1},
                        }
                    ]
                ),
            ),
            r"count must be an integer",
        ),
        (
            (
                _card(
                    effects=[
                        {
                            "trigger": "battlecry",
                            "operation": "buff",
                            "target": "self",
                            "attack": 0,
                            "health": 0,
                        }
                    ]
                ),
            ),
            r"Buff must change stats or grant a keyword",
        ),
    ],
)
def test_invalid_cards_are_reported_with_their_id(
    tmp_path: Path, cards: tuple[dict[str, object], ...], message: str
) -> None:
    with pytest.raises(CardDataError, match=message):
        load_card_library(_write(tmp_path, *cards))


def test_missing_and_malformed_card_files_are_reported(tmp_path: Path) -> None:
    with pytest.raises(CardDataError, match="does not exist"):
        load_card_library(tmp_path / "missing.json")

    broken = tmp_path / "broken.json"
    broken.write_text('{"cards": [', encoding="utf-8")
    with pytest.raises(CardDataError, match="invalid JSON at line 1"):
        load_card_library(broken)

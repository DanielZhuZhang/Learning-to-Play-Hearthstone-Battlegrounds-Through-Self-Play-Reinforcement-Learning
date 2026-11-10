import pytest

from hsbg.data.models import (
    CardEffect,
    CardEffectBinding,
    CardRecord,
    EffectTarget,
    EffectTrigger,
    IterationProfile,
    KeywordEffect,
    MinionKeyword,
)


def _card_values() -> dict[str, object]:
    return {
        "card_id": "15.6:1",
        "name": "Alleycat",
        "tier": 1,
        "tribe": "Beast",
        "card_type": "Beast",
        "attack": 1,
        "health": 1,
        "text": "Battlecry: Summon a 1/1 Cat.",
    }


def test_card_record_defaults_to_an_implemented_card_without_effects() -> None:
    card = CardRecord(**_card_values())  # type: ignore[arg-type]

    assert (card.card_id, card.name, card.attack, card.health) == ("15.6:1", "Alleycat", 1, 1)
    assert card.implemented
    assert card.effects == ()


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"card_id": ""}, "Card ID"),
        ({"tier": 0}, "tier"),
        ({"name": ""}, "Card name"),
        ({"attack": -1}, "must not be negative"),
        ({"health": -1}, "must not be negative"),
        (
            {
                "implemented": False,
                "effects": (
                    CardEffect(
                        EffectTrigger.PASSIVE,
                        KeywordEffect(EffectTarget.SELF, MinionKeyword.TAUNT),
                    ),
                ),
            },
            "Unimplemented cards must not declare effects",
        ),
    ],
)
def test_card_record_rejects_invalid_values(
    changes: dict[str, object], message: str
) -> None:
    values = _card_values()
    values.update(changes)

    with pytest.raises(ValueError, match=message):
        CardRecord(**values)  # type: ignore[arg-type]


def test_profile_accepts_unique_enabled_cards_and_their_bindings() -> None:
    profile = IterationProfile(
        iteration_id="test-15.6",
        cards_file="../cards.json",
        enabled_card_ids=("15.6:1",),
        card_effects=(CardEffectBinding("15.6:1", ()),),
    )

    assert profile.enabled_card_ids == ("15.6:1",)
    assert profile.cards_file == "../cards.json"


def test_profile_rejects_duplicate_ids_and_bindings_for_disabled_cards() -> None:
    with pytest.raises(ValueError, match="duplicates"):
        IterationProfile(
            iteration_id="test",
            cards_file="cards.json",
            enabled_card_ids=("15.6:1", "15.6:1"),
        )

    with pytest.raises(ValueError, match="enabled cards"):
        IterationProfile(
            iteration_id="test",
            cards_file="cards.json",
            enabled_card_ids=("15.6:1",),
            card_effects=(CardEffectBinding("15.6:2", ()),),
        )

    with pytest.raises(ValueError, match="cards file"):
        IterationProfile(iteration_id="test", cards_file=" ", enabled_card_ids=())

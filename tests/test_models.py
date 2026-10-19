import pytest

from hsbg.data.models import CardOverride, CardRecord, CardSource, IterationProfile


def make_card() -> CardRecord:
    return CardRecord(
        card_id="15.6:1",
        source=CardSource(workbook="season -1 (beta).xlsx", sheet="15.6", row=2),
        source_number=1,
        tier=1,
        stars="*",
        name="Alleycat",
        tribe="Beast",
        card_type="Beast",
        attack=1,
        health=1,
        text="Battlecry: Summon a 1/1 Cat.",
        source_values=(1, 1, "*", "Alleycat", "Beast", "Beast", 1, 1, "Battlecry: Summon a 1/1 Cat."),
    )


def test_card_record_keeps_normalized_and_original_source_values() -> None:
    card = make_card()

    assert card.card_id == "15.6:1"
    assert card.name == "Alleycat"
    assert card.source.sheet == "15.6"
    assert card.source.row == 2
    assert card.source_values[3] == "Alleycat"
    assert card.source_values[8] == "Battlecry: Summon a 1/1 Cat."


def test_card_source_requires_real_data_row() -> None:
    with pytest.raises(ValueError, match="data row"):
        CardSource(workbook="cards.xlsx", sheet="15.6", row=1)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"card_id": ""}, "Card ID"),
        ({"tier": 0}, "tier"),
        ({"name": ""}, "Card name"),
        ({"attack": -1}, "must not be negative"),
        ({"health": -1}, "must not be negative"),
    ],
)
def test_card_record_rejects_invalid_values(
    changes: dict[str, object], message: str
) -> None:
    values = {
        "card_id": "15.6:1",
        "source": CardSource("cards.xlsx", "15.6", 2),
        "source_number": 1,
        "tier": 1,
        "stars": "*",
        "name": "Alleycat",
        "tribe": "Beast",
        "card_type": "Beast",
        "attack": 1,
        "health": 1,
        "text": "Battlecry.",
        "source_values": (1, 1, "*", "Alleycat", "Beast", "Beast", 1, 1, "Battlecry."),
    }
    values.update(changes)

    with pytest.raises(ValueError, match=message):
        CardRecord(**values)  # type: ignore[arg-type]


def test_profile_accepts_unique_enabled_cards_and_reasoned_overrides() -> None:
    override = CardOverride(card_id="15.6:1", reason="Testing a balance change.", attack=2)
    profile = IterationProfile(
        iteration_id="test-15.6",
        workbook="season -1 (beta).xlsx",
        sheet="15.6",
        enabled_card_ids=("15.6:1",),
        overrides=(override,),
    )

    assert profile.enabled_card_ids == ("15.6:1",)
    assert profile.overrides[0].attack == 2
    assert profile.overrides[0].reason == "Testing a balance change."


def test_profile_rejects_duplicate_ids_and_overrides_for_disabled_cards() -> None:
    with pytest.raises(ValueError, match="duplicates"):
        IterationProfile(
            iteration_id="test",
            workbook="cards.xlsx",
            sheet="15.6",
            enabled_card_ids=("15.6:1", "15.6:1"),
        )

    with pytest.raises(ValueError, match="enabled cards"):
        IterationProfile(
            iteration_id="test",
            workbook="cards.xlsx",
            sheet="15.6",
            enabled_card_ids=("15.6:1",),
            overrides=(CardOverride(card_id="15.6:2", reason="Not enabled."),),
        )


def test_override_requires_reason_and_valid_stats() -> None:
    with pytest.raises(ValueError, match="reason"):
        CardOverride(card_id="15.6:1", reason=" ")

    with pytest.raises(ValueError, match="attack"):
        CardOverride(card_id="15.6:1", reason="Test override.", attack=-1)

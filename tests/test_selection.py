import pytest

from hsbg.data.models import CardOverride, CardRecord, CardSource, IterationProfile
from hsbg.data.selection import (
    ProfileSelectionError,
    apply_profile_overrides,
    select_enabled_cards,
)


def _card(card_id: str, source_number: int) -> CardRecord:
    return CardRecord(
        card_id=card_id,
        source=CardSource(workbook="cards.xlsx", sheet="15.6", row=source_number + 1),
        source_number=source_number,
        tier=1,
        stars=None,
        name=f"Card {source_number}",
        tribe=None,
        card_type=None,
        attack=1,
        health=1,
        text="",
        source_values=(),
    )


def _profile(
    card_ids: tuple[str, ...], overrides: tuple[CardOverride, ...] = ()
) -> IterationProfile:
    return IterationProfile(
        iteration_id="test",
        workbook="cards.xlsx",
        sheet="15.6",
        enabled_card_ids=card_ids,
        overrides=overrides,
    )


def test_select_enabled_cards_preserves_profile_order() -> None:
    first = _card("15.6:1", 1)
    second = _card("15.6:2", 2)

    selected = select_enabled_cards(_profile(("15.6:2", "15.6:1")), (first, second))

    assert selected == (second, first)


def test_select_enabled_cards_reports_missing_ids() -> None:
    with pytest.raises(
        ProfileSelectionError,
        match="Enabled card IDs not found.*'15.6:2'",
    ):
        select_enabled_cards(_profile(("15.6:2",)), (_card("15.6:1", 1),))


def test_select_enabled_cards_rejects_duplicate_supplied_ids() -> None:
    duplicate = _card("15.6:1", 1)

    with pytest.raises(ProfileSelectionError, match="duplicate card ID '15.6:1'"):
        select_enabled_cards(_profile(("15.6:1",)), (duplicate, duplicate))


def test_select_enabled_cards_accepts_empty_selection() -> None:
    assert select_enabled_cards(_profile(()), (_card("15.6:1", 1),)) == ()


def test_apply_profile_overrides_preserves_order_and_source_values() -> None:
    first = _card("15.6:1", 1)
    second = _card("15.6:2", 2)
    profile = _profile(
        ("15.6:2", "15.6:1"),
        (
            CardOverride(
                card_id="15.6:1",
                reason="Testing a balance change.",
                attack=3,
                health=4,
                tier=2,
                text="Updated text.",
            ),
        ),
    )

    updated = apply_profile_overrides(profile, (second, first))

    assert tuple(card.card_id for card in updated) == ("15.6:2", "15.6:1")
    assert updated[0] is second
    assert updated[1] == CardRecord(
        card_id=first.card_id,
        source=first.source,
        source_number=first.source_number,
        tier=2,
        stars=first.stars,
        name=first.name,
        tribe=first.tribe,
        card_type=first.card_type,
        attack=3,
        health=4,
        text="Updated text.",
        source_values=first.source_values,
    )
    assert first.attack == 1
    assert first.health == 1
    assert first.tier == 1
    assert first.source_values == ()


def test_apply_profile_overrides_reports_missing_target() -> None:
    profile = _profile(
        ("15.6:1", "15.6:2"),
        (CardOverride(card_id="15.6:2", reason="Test override.", attack=2),),
    )

    with pytest.raises(
        ProfileSelectionError,
        match="Override card IDs not found.*'15.6:2'",
    ):
        apply_profile_overrides(profile, (_card("15.6:1", 1),))


def test_apply_profile_overrides_reports_unsupported_fields() -> None:
    profile = _profile(
        ("15.6:1",),
        (CardOverride(card_id="15.6:1", reason="Test override.", tribe="Mech"),),
    )

    with pytest.raises(
        ProfileSelectionError,
        match="Override for '15.6:1' uses unsupported fields: tribe",
    ):
        apply_profile_overrides(profile, (_card("15.6:1", 1),))


def test_apply_profile_overrides_without_overrides_keeps_cards_unchanged() -> None:
    card = _card("15.6:1", 1)

    updated = apply_profile_overrides(_profile(("15.6:1",)), (card,))

    assert updated == (card,)
    assert updated[0] is card

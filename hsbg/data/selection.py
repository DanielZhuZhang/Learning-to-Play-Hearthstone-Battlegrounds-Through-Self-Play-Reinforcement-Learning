"""Resolve iteration profile card IDs against imported card records."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from hsbg.data.models import CardRecord, IterationProfile


class ProfileSelectionError(ValueError):
    """Raised when enabled profile cards cannot be resolved unambiguously."""


def select_enabled_cards(
    profile: IterationProfile, cards: Sequence[CardRecord]
) -> tuple[CardRecord, ...]:
    """Resolve enabled IDs in profile order without applying profile overrides."""
    cards_by_id: dict[str, CardRecord] = {}
    for card in cards:
        if card.card_id in cards_by_id:
            raise ProfileSelectionError(
                f"Supplied card records contain duplicate card ID {card.card_id!r}."
            )
        cards_by_id[card.card_id] = card

    missing_ids = [
        card_id for card_id in profile.enabled_card_ids if card_id not in cards_by_id
    ]
    if missing_ids:
        missing = ", ".join(repr(card_id) for card_id in missing_ids)
        raise ProfileSelectionError(
            f"Enabled card IDs not found in supplied card records: {missing}."
        )

    return tuple(cards_by_id[card_id] for card_id in profile.enabled_card_ids)


def apply_profile_overrides(
    profile: IterationProfile, selected_cards: Sequence[CardRecord]
) -> tuple[CardRecord, ...]:
    """Apply supported profile overrides without changing source values or order."""
    cards_by_id: dict[str, CardRecord] = {}
    for card in selected_cards:
        if card.card_id in cards_by_id:
            raise ProfileSelectionError(
                f"Selected card records contain duplicate card ID {card.card_id!r}."
            )
        cards_by_id[card.card_id] = card

    missing_ids = [
        override.card_id
        for override in profile.overrides
        if override.card_id not in cards_by_id
    ]
    if missing_ids:
        missing = ", ".join(repr(card_id) for card_id in missing_ids)
        raise ProfileSelectionError(
            f"Override card IDs not found in selected card records: {missing}."
        )

    overrides_by_id = {override.card_id: override for override in profile.overrides}
    updated: list[CardRecord] = []
    for card in selected_cards:
        override = overrides_by_id.get(card.card_id)
        if override is None:
            updated.append(card)
            continue
        if override.tribe is not None or override.card_type is not None:
            unsupported = []
            if override.tribe is not None:
                unsupported.append("tribe")
            if override.card_type is not None:
                unsupported.append("card_type")
            raise ProfileSelectionError(
                f"Override for {card.card_id!r} uses unsupported fields: "
                f"{', '.join(unsupported)}."
            )
        updated.append(
            replace(
                card,
                attack=card.attack if override.attack is None else override.attack,
                health=card.health if override.health is None else override.health,
                tier=card.tier if override.tier is None else override.tier,
                text=card.text if override.text is None else override.text,
            )
        )
    return tuple(updated)

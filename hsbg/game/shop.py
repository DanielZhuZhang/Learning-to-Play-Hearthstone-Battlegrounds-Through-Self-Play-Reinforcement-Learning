"""Shop-offer generation from the finite shared minion pool."""

from __future__ import annotations

import random
from collections.abc import Sequence

from hsbg.game.pool import MinionPool
from hsbg.game.state import ShopOffer


class ShopDrawError(ValueError):
    """Raised when shop-draw inputs are invalid."""


def draw_shop_offers(
    pool: MinionPool,
    eligible_card_ids: Sequence[str],
    offer_ids: Sequence[str],
    rng: random.Random,
) -> tuple[MinionPool, tuple[ShopOffer, ...]]:
    """Draw offers weighted by remaining copies, returning a partial shop if needed.

    Card definitions retain pool order for reproducible weighted draws. Each
    successful offer reserves one copy; offer IDs correspond to requested slots.
    """
    eligible_ids = _validate_identifiers(eligible_card_ids, "Eligible card IDs")
    requested_offer_ids = _validate_identifiers(offer_ids, "Offer IDs")
    if len(set(eligible_ids)) != len(eligible_ids):
        raise ShopDrawError("Eligible card IDs must be unique.")
    if len(set(requested_offer_ids)) != len(requested_offer_ids):
        raise ShopDrawError("Offer IDs must be unique.")
    if not isinstance(rng, random.Random):
        raise ShopDrawError("rng must be a random.Random instance.")

    pool_ids = {entry.card_id for entry in pool.entries}
    unknown_ids = set(eligible_ids) - pool_ids
    if unknown_ids:
        unknown_id = next(card_id for card_id in eligible_ids if card_id in unknown_ids)
        raise ShopDrawError(f"Eligible card ID {unknown_id!r} is not in the pool.")

    eligible_set = set(eligible_ids)
    offers: list[ShopOffer] = []
    updated_pool = pool

    for offer_id in requested_offer_ids:
        available_entries = tuple(
            entry
            for entry in updated_pool.entries
            if entry.card_id in eligible_set and entry.available_copies > 0
        )
        if not available_entries:
            break
        selected_entry = rng.choices(
            available_entries,
            weights=tuple(entry.available_copies for entry in available_entries),
            k=1,
        )[0]
        updated_pool = updated_pool.take(selected_entry.card_id)
        offers.append(
            ShopOffer(offer_id=offer_id, definition_id=selected_entry.card_id)
        )

    return updated_pool, tuple(offers)


def _validate_identifiers(values: Sequence[str], label: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ShopDrawError(f"{label} must be a sequence of non-empty strings.")
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ShopDrawError(f"{label} must contain only non-empty strings.")
    return tuple(values)

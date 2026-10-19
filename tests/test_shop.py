import random
from collections.abc import Sequence
from typing import TypeVar, cast

import pytest

from hsbg.game.pool import MinionPool, MinionPoolEntry
from hsbg.game.shop import ShopDrawError, draw_shop_offers
from hsbg.game.state import ShopOffer

T = TypeVar("T")


def _pool(*entries: tuple[str, int]) -> MinionPool:
    return MinionPool(
        entries=tuple(
            MinionPoolEntry(
                card_id=card_id,
                tier=1,
                total_copies=copies,
                available_copies=copies,
            )
            for card_id, copies in entries
        )
    )


def test_draw_shop_offers_is_seeded_and_reserves_copies() -> None:
    pool = _pool(("card-a", 3), ("card-b", 2))

    updated, offers = draw_shop_offers(
        pool,
        ("card-a", "card-b"),
        ("offer-1", "offer-2", "offer-3"),
        random.Random(42),
    )
    repeated_pool, repeated_offers = draw_shop_offers(
        pool,
        ("card-a", "card-b"),
        ("offer-1", "offer-2", "offer-3"),
        random.Random(42),
    )

    assert offers == repeated_offers
    assert updated == repeated_pool
    assert len(offers) == 3
    assert all(isinstance(offer, ShopOffer) for offer in offers)
    assert all(offer.attack_bonus == 0 and offer.health_bonus == 0 for offer in offers)
    assert updated.available_copies == pool.available_copies - len(offers)
    assert pool.available_copies == 5


def test_draw_shop_offers_uses_current_copy_counts_as_weights() -> None:
    class LastChoiceRandom(random.Random):
        def __init__(self) -> None:
            super().__init__(1)
            self.weights_seen: list[tuple[int, ...]] = []

        def choices(
            self,
            population: Sequence[T],
            weights: Sequence[float] | None = None,
            *,
            cum_weights: Sequence[float] | None = None,
            k: int = 1,
        ) -> list[T]:
            assert weights is not None
            self.weights_seen.append(tuple(int(weight) for weight in weights))
            return [population[-1]] * k

    rng = LastChoiceRandom()

    _, offers = draw_shop_offers(
        _pool(("card-a", 3), ("card-b", 1)),
        ("card-a", "card-b"),
        ("offer-1", "offer-2"),
        rng,
    )

    assert rng.weights_seen == [(3, 1), (3,)]
    assert tuple(offer.definition_id for offer in offers) == ("card-b", "card-a")


def test_draw_shop_offers_respects_available_copy_weights_and_pool_order() -> None:
    pool = MinionPool(
        entries=(
            MinionPoolEntry("out-of-tier", 2, 4, 4),
            MinionPoolEntry("eligible", 1, 3, 3),
            MinionPoolEntry("depleted", 1, 2, 0),
        )
    )

    updated, offers = draw_shop_offers(
        pool,
        ("eligible", "depleted"),
        ("offer-1", "offer-2", "offer-3", "offer-4"),
        random.Random(7),
    )

    assert offers == (
        ShopOffer("offer-1", "eligible"),
        ShopOffer("offer-2", "eligible"),
        ShopOffer("offer-3", "eligible"),
    )
    assert updated.entries[1].available_copies == 0
    assert updated.entries[0].available_copies == 4
    assert pool.entries[1].available_copies == 3


def test_draw_shop_offers_returns_partial_shop_when_pool_is_short() -> None:
    pool = MinionPool(
        entries=(
            MinionPoolEntry("card-a", tier=1, total_copies=1, available_copies=1),
            MinionPoolEntry("card-b", tier=1, total_copies=1, available_copies=0),
        )
    )

    updated, offers = draw_shop_offers(
        pool,
        ("card-a", "card-b"),
        ("offer-1", "offer-2", "offer-3"),
        random.Random(1),
    )

    assert offers == (ShopOffer("offer-1", "card-a"),)
    assert updated.entries[0].available_copies == 0
    assert updated.entries[1].available_copies == 0
    assert updated.available_copies == 0


def test_draw_shop_offers_with_no_eligible_cards_returns_empty_shop() -> None:
    pool = _pool(("card-a", 2))

    updated, offers = draw_shop_offers(
        pool, (), ("offer-1", "offer-2"), random.Random(1)
    )

    assert updated is pool
    assert offers == ()


@pytest.mark.parametrize(
    ("eligible_card_ids", "offer_ids", "message"),
    [
        (("missing",), ("offer-1",), "not in the pool"),
        (("card-a", "card-a"), ("offer-1",), "Eligible card IDs must be unique"),
        (("card-a",), ("offer-1", "offer-1"), "Offer IDs must be unique"),
        (("card-a",), ("",), "Offer IDs must contain only non-empty strings"),
    ],
)
def test_draw_shop_offers_rejects_invalid_identifiers(
    eligible_card_ids: tuple[str, ...],
    offer_ids: tuple[str, ...],
    message: str,
) -> None:
    with pytest.raises(ShopDrawError, match=message):
        draw_shop_offers(
            _pool(("card-a", 1)),
            eligible_card_ids,
            offer_ids,
            random.Random(1),
        )


def test_draw_shop_offers_rejects_non_random_generator() -> None:
    with pytest.raises(ShopDrawError, match="random.Random"):
        draw_shop_offers(
            _pool(("card-a", 1)),
            ("card-a",),
            (),
            cast(random.Random, object()),
        )

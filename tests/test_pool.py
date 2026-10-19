from dataclasses import replace
from pathlib import Path

import pytest

from hsbg.data.models import CardRecord
from hsbg.data.profiles import load_profile
from hsbg.data.selection import select_enabled_cards
from hsbg.data.workbook import load_sheet
from hsbg.game.config import GameRules
from hsbg.game.pool import (
    MinionPool,
    MinionPoolEntry,
    PoolAccountingError,
    PoolConfigurationError,
    build_initial_pool,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _selected_cards() -> tuple[CardRecord, ...]:
    profile = load_profile(PROJECT_ROOT / "data" / "profiles" / "initial-15.6.json")
    return select_enabled_cards(
        profile,
        load_sheet(PROJECT_ROOT / profile.workbook, profile.sheet),
    )


def test_initial_pool_has_one_entry_per_enabled_card() -> None:
    profile_cards = _selected_cards()
    rules = GameRules()

    pool = build_initial_pool(profile_cards, rules)

    assert len(pool.entries) == 29
    assert tuple(entry.card_id for entry in pool.entries) == tuple(
        card.card_id for card in profile_cards
    )
    assert pool.total_copies == 382
    assert pool.available_copies == 382
    assert {
        tier: sum(
            entry.available_copies for entry in pool.entries if entry.tier == tier
        )
        for tier in range(1, 7)
    } == {1: 112, 2: 120, 3: 78, 4: 22, 5: 36, 6: 14}


def test_pool_copy_count_is_per_card_and_does_not_limit_unique_entries() -> None:
    pool = build_initial_pool(_selected_cards(), GameRules())

    tier_one_entries = [entry for entry in pool.entries if entry.tier == 1]

    assert len(tier_one_entries) == 7
    assert {entry.available_copies for entry in tier_one_entries} == {16}


def test_initial_pool_rejects_duplicate_card_ids() -> None:
    card = _selected_cards()[0]

    with pytest.raises(PoolConfigurationError, match="duplicate card ID"):
        build_initial_pool((card, card), GameRules())


def test_initial_pool_rejects_card_outside_configured_tier_range() -> None:
    card = replace(_selected_cards()[0], tier=7)

    with pytest.raises(PoolConfigurationError, match="outside configured Tavern tiers"):
        build_initial_pool((card,), GameRules())


def test_pool_model_rejects_duplicate_definition_entries() -> None:
    entry = MinionPoolEntry("15.6:1", tier=1, total_copies=16, available_copies=16)

    with pytest.raises(ValueError, match="Pool card IDs must be unique"):
        MinionPool(entries=(entry, entry))


def test_taking_a_copy_returns_new_pool_and_preserves_original() -> None:
    pool = build_initial_pool(_selected_cards(), GameRules())

    updated = pool.take("15.6:1")

    assert pool.entries[0].available_copies == 16
    assert updated.entries[0].available_copies == 15
    assert updated.entries[0].total_copies == 16
    assert updated.available_copies == pool.available_copies - 1
    assert tuple(entry.card_id for entry in updated.entries) == tuple(
        entry.card_id for entry in pool.entries
    )


def test_returning_a_copy_restores_availability_without_mutating_original() -> None:
    pool = build_initial_pool(_selected_cards(), GameRules())
    depleted = pool.take("15.6:1")

    restored = depleted.return_copy("15.6:1")

    assert depleted.entries[0].available_copies == 15
    assert restored.entries[0].available_copies == 16
    assert restored.total_copies == pool.total_copies
    assert restored.available_copies == pool.available_copies


def test_pool_accounting_rejects_unknown_and_exhausted_cards() -> None:
    pool = build_initial_pool(
        _selected_cards()[:1],
        replace(GameRules(), pool_copies_by_tier=(1, 15, 13, 11, 9, 7)),
    )

    with pytest.raises(PoolAccountingError, match="not in the pool"):
        pool.take("15.6:2")

    depleted = pool.take("15.6:1")
    with pytest.raises(PoolAccountingError, match="No available copies remain"):
        depleted.take("15.6:1")


def test_pool_accounting_rejects_returns_above_configured_total() -> None:
    pool = build_initial_pool(_selected_cards()[:1], GameRules())

    with pytest.raises(PoolAccountingError, match="exceed its configured 16 copies"):
        pool.return_copy("15.6:1")

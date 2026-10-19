"""Initial per-card copy inventory for a finite shared minion pool."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from hsbg.data.models import CardRecord
from hsbg.game.config import GameRules


class PoolConfigurationError(ValueError):
    """Raised when selected cards cannot be represented by the pool rules."""


class PoolAccountingError(ValueError):
    """Raised when a pool copy cannot be taken or returned legally."""


@dataclass(frozen=True, slots=True)
class MinionPoolEntry:
    """Configured and currently available copies of one card definition."""

    card_id: str
    tier: int
    total_copies: int
    available_copies: int

    def __post_init__(self) -> None:
        if not isinstance(self.card_id, str) or not self.card_id.strip():
            raise ValueError("Pool entry card ID must not be empty.")
        if (
            isinstance(self.tier, bool)
            or not isinstance(self.tier, int)
            or self.tier < 1
        ):
            raise ValueError("Pool entry tier must be a positive integer.")
        if (
            isinstance(self.total_copies, bool)
            or not isinstance(self.total_copies, int)
            or self.total_copies < 1
        ):
            raise ValueError("Pool entry total copies must be a positive integer.")
        if (
            isinstance(self.available_copies, bool)
            or not isinstance(self.available_copies, int)
            or self.available_copies < 0
        ):
            raise ValueError("Pool entry available copies must be nonnegative.")
        if self.available_copies > self.total_copies:
            raise ValueError("Available copies must not exceed total copies.")


@dataclass(frozen=True, slots=True)
class MinionPool:
    """Immutable initial shared-pool inventory, one entry per card ID."""

    entries: tuple[MinionPoolEntry, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.entries, tuple) or any(
            not isinstance(entry, MinionPoolEntry) for entry in self.entries
        ):
            raise ValueError("Pool entries must be a tuple of MinionPoolEntry values.")
        card_ids = [entry.card_id for entry in self.entries]
        if len(set(card_ids)) != len(card_ids):
            raise ValueError("Pool card IDs must be unique.")

    @property
    def total_copies(self) -> int:
        """Total configured copies across all enabled card definitions."""
        return sum(entry.total_copies for entry in self.entries)

    @property
    def available_copies(self) -> int:
        """Total currently available copies across all definitions."""
        return sum(entry.available_copies for entry in self.entries)

    def take(self, card_id: str) -> MinionPool:
        """Return a new pool with one available copy reserved."""
        entry = self._entry(card_id)
        if entry.available_copies == 0:
            raise PoolAccountingError(
                f"No available copies remain for card {card_id!r}."
            )
        return self._replace_entry(
            replace(entry, available_copies=entry.available_copies - 1)
        )

    def return_copy(self, card_id: str) -> MinionPool:
        """Return a copy, without exceeding the configured supply."""
        entry = self._entry(card_id)
        if entry.available_copies == entry.total_copies:
            raise PoolAccountingError(
                f"Returning card {card_id!r} would exceed its configured "
                f"{entry.total_copies} copies."
            )
        return self._replace_entry(
            replace(entry, available_copies=entry.available_copies + 1)
        )

    def _entry(self, card_id: str) -> MinionPoolEntry:
        for entry in self.entries:
            if entry.card_id == card_id:
                return entry
        raise PoolAccountingError(f"Card ID {card_id!r} is not in the pool.")

    def _replace_entry(self, replacement: MinionPoolEntry) -> MinionPool:
        return MinionPool(
            entries=tuple(
                replacement if entry.card_id == replacement.card_id else entry
                for entry in self.entries
            )
        )


def build_initial_pool(
    selected_cards: Sequence[CardRecord], rules: GameRules
) -> MinionPool:
    """Create one pool entry per selected card using its tier's copy count."""
    entries: list[MinionPoolEntry] = []
    seen_ids: set[str] = set()
    for card in selected_cards:
        if card.card_id in seen_ids:
            raise PoolConfigurationError(
                f"Selected cards contain duplicate card ID {card.card_id!r}."
            )
        seen_ids.add(card.card_id)
        if card.tier > rules.max_tavern_tier:
            raise PoolConfigurationError(
                f"Card {card.card_id!r} has tier {card.tier}, outside configured "
                f"Tavern tiers 1–{rules.max_tavern_tier}."
            )
        entries.append(
            MinionPoolEntry(
                card_id=card.card_id,
                tier=card.tier,
                total_copies=rules.pool_copies_by_tier[card.tier - 1],
                available_copies=rules.pool_copies_by_tier[card.tier - 1],
            )
        )
    return MinionPool(entries=tuple(entries))

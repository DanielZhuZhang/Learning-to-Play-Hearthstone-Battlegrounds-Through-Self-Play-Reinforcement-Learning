"""Validated match-state value models; no state transitions are performed here."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from hsbg.data.models import MinionKeyword, TokenSpec


class MinionZone(StrEnum):
    """Owned zones in which a minion instance can exist."""

    HAND = "hand"
    BOARD = "board"


class MatchPhase(StrEnum):
    """High-level phases represented by match state."""

    RECRUIT = "recruit"
    COMBAT = "combat"
    FINISHED = "finished"


@dataclass(frozen=True, slots=True)
class ShopOffer:
    """A shop slot referencing a definition, not an owned minion copy."""

    offer_id: str
    definition_id: str
    attack_bonus: int = 0
    health_bonus: int = 0
    granted_keywords: tuple[MinionKeyword, ...] = ()

    def __post_init__(self) -> None:
        _validate_identifier(self.offer_id, "Shop offer ID")
        _validate_identifier(self.definition_id, "Shop offer definition ID")
        _validate_modifiers(self.attack_bonus, self.health_bonus)
        _validate_keywords(self.granted_keywords)


@dataclass(frozen=True, slots=True)
class MinionInstance:
    """One owned card/token copy, distinct from its shared definition."""

    instance_id: str
    definition_id: str
    zone: MinionZone
    attack_bonus: int = 0
    health_bonus: int = 0
    granted_keywords: tuple[MinionKeyword, ...] = ()
    pool_owned: bool = True
    token_spec: TokenSpec | None = None

    def __post_init__(self) -> None:
        _validate_identifier(self.instance_id, "Minion instance ID")
        _validate_identifier(self.definition_id, "Minion instance definition ID")
        if not isinstance(self.zone, MinionZone):
            raise ValueError("Minion zone must be a MinionZone value.")
        if not isinstance(self.pool_owned, bool):
            raise ValueError("pool_owned must be a boolean.")
        if self.token_spec is not None and not isinstance(self.token_spec, TokenSpec):
            raise ValueError("token_spec must be a TokenSpec value or None.")
        if self.pool_owned and self.token_spec is not None:
            raise ValueError("A minion cannot be both pool-owned and a token.")
        _validate_modifiers(self.attack_bonus, self.health_bonus)
        _validate_keywords(self.granted_keywords)


@dataclass(frozen=True, slots=True)
class PlayerState:
    """Player resources, shop offers, and owned minion instances."""

    player_id: str
    health: int
    armor: int
    gold: int
    tavern_tier: int
    minions: tuple[MinionInstance, ...] = ()
    shop: tuple[ShopOffer, ...] = ()
    shop_frozen: bool = False

    def __post_init__(self) -> None:
        _validate_identifier(self.player_id, "Player ID")
        for name, value in (
            ("health", self.health),
            ("armor", self.armor),
            ("gold", self.gold),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer.")
        if (
            isinstance(self.tavern_tier, bool)
            or not isinstance(self.tavern_tier, int)
            or self.tavern_tier < 1
        ):
            raise ValueError("tavern_tier must be a positive integer.")
        if not isinstance(self.shop_frozen, bool):
            raise ValueError("shop_frozen must be a boolean.")
        if not isinstance(self.minions, tuple) or any(
            not isinstance(minion, MinionInstance) for minion in self.minions
        ):
            raise ValueError("Player minions must be a tuple of MinionInstance values.")
        if not isinstance(self.shop, tuple) or any(
            not isinstance(offer, ShopOffer) for offer in self.shop
        ):
            raise ValueError("Player shop must be a tuple of ShopOffer values.")

        instance_ids = [minion.instance_id for minion in self.minions]
        if len(set(instance_ids)) != len(instance_ids):
            raise ValueError("Owned minion instance IDs must be unique per player.")
        offer_ids = [offer.offer_id for offer in self.shop]
        if len(set(offer_ids)) != len(offer_ids):
            raise ValueError("Shop offer IDs must be unique per player.")

    @property
    def hand(self) -> tuple[MinionInstance, ...]:
        """Owned instances currently in hand, in stored order."""
        return tuple(
            minion for minion in self.minions if minion.zone is MinionZone.HAND
        )

    @property
    def board(self) -> tuple[MinionInstance, ...]:
        """Owned instances currently on board, in stored order."""
        return tuple(
            minion for minion in self.minions if minion.zone is MinionZone.BOARD
        )


@dataclass(frozen=True, slots=True)
class MatchState:
    """High-level match snapshot with validated player identity and phase."""

    turn: int
    phase: MatchPhase
    players: tuple[PlayerState, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.turn, bool)
            or not isinstance(self.turn, int)
            or self.turn < 1
        ):
            raise ValueError("Match turn must be a positive integer.")
        if not isinstance(self.phase, MatchPhase):
            raise ValueError("Match phase must be a MatchPhase value.")
        if not isinstance(self.players, tuple) or any(
            not isinstance(player, PlayerState) for player in self.players
        ):
            raise ValueError("Match players must be a tuple of PlayerState values.")
        if len(self.players) < 2:
            raise ValueError("A match must contain at least two players.")
        player_ids = [player.player_id for player in self.players]
        if len(set(player_ids)) != len(player_ids):
            raise ValueError("Player IDs must be unique within a match.")
        instance_ids = [
            minion.instance_id for player in self.players for minion in player.minions
        ]
        if len(set(instance_ids)) != len(instance_ids):
            raise ValueError("Minion instance IDs must be unique within a match.")


def _validate_identifier(value: str, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must not be empty.")


def _validate_modifiers(attack_bonus: int, health_bonus: int) -> None:
    for name, value in (
        ("attack_bonus", attack_bonus),
        ("health_bonus", health_bonus),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer.")


def _validate_keywords(keywords: tuple[MinionKeyword, ...]) -> None:
    if not isinstance(keywords, tuple) or any(
        not isinstance(keyword, MinionKeyword) for keyword in keywords
    ):
        raise ValueError("Granted keywords must be a tuple of MinionKeyword values.")
    if len(set(keywords)) != len(keywords):
        raise ValueError("Granted keywords must not contain duplicates.")

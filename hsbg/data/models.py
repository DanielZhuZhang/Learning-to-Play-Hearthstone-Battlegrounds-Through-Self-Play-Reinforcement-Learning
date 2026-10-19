"""Typed models for source card data and explicit iteration selections."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from hsbg.game.config import GameRules

CellValue = str | int | float | bool | None


class EffectTrigger(StrEnum):
    """Event or state under which a declared card effect applies."""

    PASSIVE = "passive"
    BATTLECRY = "battlecry"
    DEATHRATTLE = "deathrattle"


class EffectTarget(StrEnum):
    """Supported deterministic recipient selectors for declared effects."""

    SELF = "self"
    FRIENDLY_MURLOC = "friendly_murloc"
    FRIENDLY_DEMON = "friendly_demon"
    OTHER_FRIENDLY_MECHS = "other_friendly_mechs"
    FRIENDLY_BEAST = "friendly_beast"
    FRIENDLY_DEMONS = "friendly_demons"
    FRIENDLY_MECH = "friendly_mech"
    OTHER_FRIENDLY_MURLOCS = "other_friendly_murlocs"
    FRIENDLY_TAUNT_MINIONS = "friendly_taunt_minions"
    ALL_FRIENDLY_MINIONS = "all_friendly_minions"


class MinionKeyword(StrEnum):
    """Combat keywords represented by the current selected roster."""

    TAUNT = "taunt"
    DIVINE_SHIELD = "divine_shield"
    POISONOUS = "poisonous"


@dataclass(frozen=True, slots=True)
class KeywordEffect:
    """A keyword attached to a card or granted to a selected target."""

    target: EffectTarget
    keyword: MinionKeyword


@dataclass(frozen=True, slots=True)
class TokenSpec:
    """Explicit token stats used by a summon effect."""

    name: str
    attack: int
    health: int
    keywords: tuple[MinionKeyword, ...] = ()

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Summoned token name must not be empty.")
        if isinstance(self.attack, bool) or not isinstance(self.attack, int):
            raise ValueError("Summoned token attack must be an integer.")
        if self.attack < 0:
            raise ValueError("Summoned token attack must not be negative.")
        if isinstance(self.health, bool) or not isinstance(self.health, int):
            raise ValueError("Summoned token health must be an integer.")
        if self.health < 0:
            raise ValueError("Summoned token health must not be negative.")


@dataclass(frozen=True, slots=True)
class SummonEffect:
    """A fixed-count summon of one explicitly described token."""

    token: TokenSpec
    count: int

    def __post_init__(self) -> None:
        if isinstance(self.count, bool) or not isinstance(self.count, int):
            raise ValueError("Summon count must be an integer.")
        if self.count < 1:
            raise ValueError("Summon count must be a positive integer.")


@dataclass(frozen=True, slots=True)
class BuffEffect:
    """A deterministic stat/keyword change for cards matching a selector."""

    target: EffectTarget
    attack: int = 0
    health: int = 0
    keywords: tuple[MinionKeyword, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.attack, bool) or not isinstance(self.attack, int):
            raise ValueError("Buff attack must be an integer.")
        if self.attack < 0:
            raise ValueError("Buff attack must not be negative.")
        if isinstance(self.health, bool) or not isinstance(self.health, int):
            raise ValueError("Buff health must be an integer.")
        if self.health < 0:
            raise ValueError("Buff health must not be negative.")
        if self.attack == 0 and self.health == 0 and not self.keywords:
            raise ValueError("Buff must change stats or grant a keyword.")


@dataclass(frozen=True, slots=True)
class DamageHeroEffect:
    """Damage dealt directly to the owning player's hero."""

    amount: int

    def __post_init__(self) -> None:
        if isinstance(self.amount, bool) or not isinstance(self.amount, int):
            raise ValueError("Hero damage must be an integer.")
        if self.amount < 1:
            raise ValueError("Hero damage must be a positive integer.")


EffectOperation = KeywordEffect | SummonEffect | BuffEffect | DamageHeroEffect


@dataclass(frozen=True, slots=True)
class CardEffect:
    """A machine-readable effect declaration; it does not execute the effect."""

    trigger: EffectTrigger
    operation: EffectOperation

    def __post_init__(self) -> None:
        is_passive_keyword = self.trigger is EffectTrigger.PASSIVE and isinstance(
            self.operation, KeywordEffect
        )
        if not is_passive_keyword:
            if self.trigger is EffectTrigger.PASSIVE:
                raise ValueError("Passive effects must declare a keyword.")
            if isinstance(self.operation, KeywordEffect):
                raise ValueError("Keyword effects must use the passive trigger.")


@dataclass(frozen=True, slots=True)
class CardEffectBinding:
    """Effect declarations for one selected card, separate from source text."""

    card_id: str
    effects: tuple[CardEffect, ...]

    def __post_init__(self) -> None:
        if not self.card_id.strip():
            raise ValueError("Effect binding card ID must not be empty.")


@dataclass(frozen=True, slots=True)
class CardSource:
    """Location of a card record in its source workbook."""

    workbook: str
    sheet: str
    row: int

    def __post_init__(self) -> None:
        if not self.workbook.strip():
            raise ValueError("Card source workbook must not be empty.")
        if not self.sheet.strip():
            raise ValueError("Card source sheet must not be empty.")
        if self.row < 2:
            raise ValueError(
                "Card source row must refer to a data row (row 2 or later)."
            )


@dataclass(frozen=True, slots=True)
class CardRecord:
    """Normalized card fields alongside the unchanged source worksheet values."""

    card_id: str
    source: CardSource
    source_number: int
    tier: int
    stars: str | None
    name: str
    tribe: str | None
    card_type: str | None
    attack: int
    health: int
    text: str
    source_values: tuple[CellValue, ...]

    def __post_init__(self) -> None:
        if not self.card_id.strip():
            raise ValueError("Card ID must not be empty.")
        if self.source_number < 1:
            raise ValueError("Card source number must be positive.")
        if self.tier < 1:
            raise ValueError("Card tier must be positive.")
        if not self.name.strip():
            raise ValueError("Card name must not be empty.")
        if self.attack < 0 or self.health < 0:
            raise ValueError("Base attack and health must not be negative.")


@dataclass(frozen=True, slots=True)
class CardOverride:
    """An intentional profile-specific replacement for selected card fields."""

    card_id: str
    reason: str
    tier: int | None = None
    tribe: str | None = None
    card_type: str | None = None
    attack: int | None = None
    health: int | None = None
    text: str | None = None

    def __post_init__(self) -> None:
        if not self.card_id.strip():
            raise ValueError("Override card ID must not be empty.")
        if not self.reason.strip():
            raise ValueError("Every card override must include a reason.")
        if self.tier is not None and self.tier < 1:
            raise ValueError("Override tier must be positive.")
        if self.attack is not None and self.attack < 0:
            raise ValueError("Override attack must not be negative.")
        if self.health is not None and self.health < 0:
            raise ValueError("Override health must not be negative.")


@dataclass(frozen=True, slots=True)
class IterationProfile:
    """Workbook source, enabled card IDs, and explicit team-owned overrides."""

    iteration_id: str
    workbook: str
    sheet: str
    enabled_card_ids: tuple[str, ...]
    overrides: tuple[CardOverride, ...] = ()
    card_effects: tuple[CardEffectBinding, ...] = ()
    rules: GameRules = GameRules()

    def __post_init__(self) -> None:
        if not self.iteration_id.strip():
            raise ValueError("Iteration ID must not be empty.")
        if not self.workbook.strip():
            raise ValueError("Iteration workbook must not be empty.")
        if not self.sheet.strip():
            raise ValueError("Iteration sheet must not be empty.")
        if len(set(self.enabled_card_ids)) != len(self.enabled_card_ids):
            raise ValueError("Enabled card IDs must not contain duplicates.")
        if any(not card_id.strip() for card_id in self.enabled_card_ids):
            raise ValueError("Enabled card IDs must not be empty.")
        override_ids = [override.card_id for override in self.overrides]
        if len(set(override_ids)) != len(override_ids):
            raise ValueError("A profile may override each card ID at most once.")
        unknown_override_ids = set(override_ids) - set(self.enabled_card_ids)
        if unknown_override_ids:
            unknown = ", ".join(sorted(unknown_override_ids))
            raise ValueError(
                f"Overrides must target enabled cards; not enabled: {unknown}."
            )
        effect_ids = [binding.card_id for binding in self.card_effects]
        if len(set(effect_ids)) != len(effect_ids):
            raise ValueError(
                "A profile may declare effects for each card ID at most once."
            )
        unknown_effect_ids = set(effect_ids) - set(self.enabled_card_ids)
        if unknown_effect_ids:
            unknown = ", ".join(sorted(unknown_effect_ids))
            raise ValueError(
                f"Effect declarations must target enabled cards; not enabled: {unknown}."
            )

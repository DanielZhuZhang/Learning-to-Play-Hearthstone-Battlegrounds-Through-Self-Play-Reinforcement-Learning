"""Execution of selected deterministic effects during the recruit phase."""

from __future__ import annotations

from dataclasses import replace
from collections.abc import Sequence

from hsbg.data.models import (
    BuffEffect,
    CardEffectBinding,
    CardRecord,
    DamageHeroEffect,
    EffectTarget,
    EffectTrigger,
    KeywordEffect,
    MinionKeyword,
    SummonEffect,
)
from hsbg.game.config import GameRules
from hsbg.game.state import MinionInstance, MinionZone, PlayerState
from hsbg.game.actions import play_minion


class EffectResolutionError(ValueError):
    """Raised when an effect cannot be resolved from the supplied profile data."""


def resolve_battlecry(
    player: PlayerState,
    source_instance_id: str,
    cards: Sequence[CardRecord],
    effect_bindings: Sequence[CardEffectBinding],
    rules: GameRules,
) -> PlayerState:
    """Apply one played minion's declared Battlecry effects in profile order."""
    source = next(
        (minion for minion in player.board if minion.instance_id == source_instance_id),
        None,
    )
    if source is None:
        raise EffectResolutionError(
            f"Battlecry source {source_instance_id!r} is not on the board."
        )

    card = next(
        (entry for entry in cards if entry.card_id == source.definition_id), None
    )
    if card is None:
        raise EffectResolutionError(
            f"Card definition {source.definition_id!r} is missing from selected cards."
        )
    binding = next(
        (entry for entry in effect_bindings if entry.card_id == card.card_id), None
    )
    if binding is None:
        raise EffectResolutionError(
            f"Card {card.card_id!r} has no structured effect declaration."
        )

    updated = player
    for effect_index, effect in enumerate(binding.effects):
        if effect.trigger is not EffectTrigger.BATTLECRY:
            continue
        operation = effect.operation
        if isinstance(operation, SummonEffect):
            updated = _summon_tokens(
                updated, operation, source.instance_id, effect_index, rules
            )
        elif isinstance(operation, BuffEffect):
            updated = _apply_buff(
                updated,
                operation,
                source.instance_id,
                cards,
                effect_bindings,
            )
        elif isinstance(operation, DamageHeroEffect):
            updated = _damage_hero(updated, operation.amount)
        elif isinstance(operation, KeywordEffect):
            raise EffectResolutionError(
                "Keyword effects are passive and cannot execute as Battlecries."
            )
        else:
            raise EffectResolutionError(
                f"Unsupported Battlecry operation {type(operation).__name__}."
            )
    return updated


def _summon_tokens(
    player: PlayerState,
    effect: SummonEffect,
    source_instance_id: str,
    effect_index: int,
    rules: GameRules,
) -> PlayerState:
    free_slots = max(0, rules.board_limit - len(player.board))
    count = min(effect.count, free_slots)
    if count == 0:
        return player

    tokens = tuple(
        MinionInstance(
            instance_id=f"{source_instance_id}:token:{effect_index}:{index}",
            definition_id=f"token:{source_instance_id}:{effect_index}:{index}",
            zone=MinionZone.BOARD,
            pool_owned=False,
            token_spec=effect.token,
        )
        for index in range(count)
    )
    existing_ids = {minion.instance_id for minion in player.minions}
    if any(token.instance_id in existing_ids for token in tokens):
        raise EffectResolutionError("Generated token instance ID already exists.")
    return replace(player, minions=player.minions + tokens)


def _apply_buff(
    player: PlayerState,
    effect: BuffEffect,
    source_instance_id: str,
    cards: Sequence[CardRecord],
    effect_bindings: Sequence[CardEffectBinding],
) -> PlayerState:
    cards_by_id = {card.card_id: card for card in cards}
    bindings_by_id = {binding.card_id: binding for binding in effect_bindings}
    instances_by_id = {minion.instance_id: minion for minion in player.board}
    matching_ids = tuple(
        minion.instance_id
        for minion in player.board
        if _matches_target(
            effect.target,
            minion,
            cards_by_id,
            bindings_by_id,
            source_instance_id,
        )
    )
    changed = {
        instance_id: replace(
            instances_by_id[instance_id],
            attack_bonus=instances_by_id[instance_id].attack_bonus + effect.attack,
            health_bonus=instances_by_id[instance_id].health_bonus + effect.health,
            granted_keywords=tuple(
                dict.fromkeys(
                    instances_by_id[instance_id].granted_keywords + effect.keywords
                )
            ),
        )
        for instance_id in matching_ids
    }
    return replace(
        player,
        minions=tuple(
            changed.get(minion.instance_id, minion) for minion in player.minions
        ),
    )


def _matches_target(
    target: EffectTarget,
    minion: MinionInstance,
    cards_by_id: dict[str, CardRecord],
    bindings_by_id: dict[str, CardEffectBinding],
    source_instance_id: str,
) -> bool:
    card = cards_by_id.get(minion.definition_id)
    if target is EffectTarget.ALL_FRIENDLY_MINIONS:
        return True
    if target is EffectTarget.SELF:
        return minion.instance_id == source_instance_id
    if target is EffectTarget.OTHER_FRIENDLY_MECHS:
        return (
            minion.instance_id != source_instance_id
            and card is not None
            and card.tribe == "Mech"
        )
    if target is EffectTarget.OTHER_FRIENDLY_MURLOCS:
        return (
            minion.instance_id != source_instance_id
            and card is not None
            and card.tribe == "Murloc"
        )
    if target is EffectTarget.FRIENDLY_TAUNT_MINIONS:
        return _has_keyword(
            minion,
            card,
            bindings_by_id.get(card.card_id) if card is not None else None,
            MinionKeyword.TAUNT,
        )
    tribe_by_target = {
        EffectTarget.FRIENDLY_MURLOC: "Murloc",
        EffectTarget.FRIENDLY_DEMON: "Demon",
        EffectTarget.FRIENDLY_DEMONS: "Demon",
        EffectTarget.FRIENDLY_MECH: "Mech",
        EffectTarget.FRIENDLY_BEAST: "Beast",
    }
    tribe = tribe_by_target.get(target)
    if tribe is None:
        raise EffectResolutionError(f"Unsupported effect target {target.value!r}.")
    return card is not None and (card.tribe == tribe or card.card_type == "All")


def _has_keyword(
    minion: MinionInstance,
    card: CardRecord | None,
    binding: CardEffectBinding | None,
    keyword: MinionKeyword,
) -> bool:
    if keyword in minion.granted_keywords:
        return True
    if card is None or binding is None:
        return False
    return any(
        effect.trigger is EffectTrigger.PASSIVE
        and isinstance(effect.operation, KeywordEffect)
        and effect.operation.target is EffectTarget.SELF
        and effect.operation.keyword is keyword
        for effect in binding.effects
    )


def _damage_hero(player: PlayerState, amount: int) -> PlayerState:
    armor_damage = min(player.armor, amount)
    remaining_damage = amount - armor_damage
    return replace(
        player,
        armor=player.armor - armor_damage,
        health=max(0, player.health - remaining_damage),
    )


def play_minion_and_resolve_battlecry(
    player: PlayerState,
    instance_id: str,
    cards: Sequence[CardRecord],
    effect_bindings: Sequence[CardEffectBinding],
    rules: GameRules,
) -> PlayerState:
    """Play a minion to the board, then execute its declared Battlecry."""
    played = play_minion(player, instance_id, rules)
    return resolve_battlecry(played, instance_id, cards, effect_bindings, rules)

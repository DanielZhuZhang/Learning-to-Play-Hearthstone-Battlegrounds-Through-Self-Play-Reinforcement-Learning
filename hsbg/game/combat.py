"""Seeded combat resolution for the effects enabled by an iteration profile."""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from hsbg.data.models import (
    BuffEffect,
    CardEffectBinding,
    CardRecord,
    EffectTarget,
    EffectTrigger,
    KeywordEffect,
    MinionKeyword,
    SummonEffect,
)
from hsbg.game.config import GameRules
from hsbg.game.effects import EffectResolutionError
from hsbg.game.state import PlayerState


@dataclass(frozen=True, slots=True)
class CombatEvent:
    """One auditable combat event, in resolution order."""

    kind: str
    source_instance_id: str
    target_instance_id: str | None = None
    amount: int = 0


@dataclass(frozen=True, slots=True)
class CombatMinionResult:
    """Final combat-only state for one surviving minion."""

    instance_id: str
    definition_id: str
    attack: int
    health: int
    max_health: int
    keywords: tuple[MinionKeyword, ...]
    token: bool
    token_name: str | None = None


@dataclass(frozen=True, slots=True)
class CombatFrame:
    """Both boards after one combat step, for step-by-step visualisation.

    ``boards`` follows the ``resolve_combat`` argument order. Minions killed in
    the step are still present (with health at or below zero) in ``hit`` frames
    and are removed in the following ``deaths`` frame.
    """

    kind: str
    source_instance_id: str | None
    target_instance_id: str | None
    boards: tuple[tuple[CombatMinionResult, ...], tuple[CombatMinionResult, ...]]


@dataclass(frozen=True, slots=True)
class CombatResult:
    """Outcome and ordered events from one combat without mutating boards."""

    winner_id: str | None
    loser_id: str | None
    damage: int
    attacks: int
    first_attacker_id: str | None
    survivors_by_player: tuple[tuple[str, tuple[CombatMinionResult, ...]], ...]
    events: tuple[CombatEvent, ...]
    frames: tuple[CombatFrame, ...] = ()


@dataclass(slots=True)
class _CombatMinion:
    instance_id: str
    definition_id: str
    attack: int
    health: int
    max_health: int
    keywords: list[MinionKeyword]
    token: bool
    dead: bool = False
    token_name: str | None = None


def resolve_combat(
    first_player: PlayerState,
    second_player: PlayerState,
    cards: Sequence[CardRecord],
    effect_bindings: Sequence[CardEffectBinding],
    rng: random.Random,
    rules: GameRules,
    *,
    record_frames: bool = False,
) -> CombatResult:
    """Resolve alternating attacks, triggers, deaths, and survivor damage.

    When board sizes tie, the supplied RNG chooses initiative. Attack damage is
    simultaneous; simultaneous Deathrattles resolve left-to-right for the
    player with initiative, then left-to-right for the other player. Tokens
    summoned during combat cannot exceed the configured board limit.

    With ``record_frames`` the result also carries a board snapshot after each
    step; recording never consumes randomness, so outcomes are unchanged.
    """
    if first_player.player_id == second_player.player_id:
        raise ValueError("Combat players must have distinct IDs.")
    if not isinstance(rng, random.Random):
        raise ValueError("rng must be a random.Random instance.")

    cards_by_id = {card.card_id: card for card in cards}
    bindings_by_id = {binding.card_id: binding for binding in effect_bindings}
    boards = [
        _build_board(first_player, cards_by_id, bindings_by_id),
        _build_board(second_player, cards_by_id, bindings_by_id),
    ]
    players = (first_player, second_player)
    events: list[CombatEvent] = []
    frames: list[CombatFrame] = []

    def record(kind: str, source: str | None = None, target: str | None = None) -> None:
        if record_frames:
            frames.append(CombatFrame(kind, source, target, _snapshot(boards)))

    if len(boards[0]) == len(boards[1]):
        active = rng.randrange(2)
    else:
        active = 0 if len(boards[0]) > len(boards[1]) else 1
    first_attacker_id = players[active].player_id
    cursors = [0, 0]
    attacks = 0
    record("start")

    while boards[0] and boards[1]:
        attacker_side = active
        defender_side = 1 - active
        attacker_index = _next_alive_index(
            boards[attacker_side], cursors[attacker_side]
        )
        if attacker_index is None:
            if not any(not minion.dead for minion in boards[attacker_side]):
                break
            cursors[attacker_side] = 0
            continue
        attacker = boards[attacker_side][attacker_index]
        target = _choose_target(boards[defender_side], rng)
        if target is None:
            break
        cursors[attacker_side] = attacker_index + 1
        attacks += 1
        events.append(CombatEvent("attack", attacker.instance_id, target.instance_id))
        record("attack", attacker.instance_id, target.instance_id)
        attacker_had_shield = MinionKeyword.DIVINE_SHIELD in attacker.keywords
        target_had_shield = MinionKeyword.DIVINE_SHIELD in target.keywords
        progress = _deal_attack_damage(attacker, target, events)
        progress |= _deal_attack_damage(target, attacker, events)
        if (
            not target_had_shield
            and MinionKeyword.POISONOUS in attacker.keywords
            and attacker.attack > 0
        ):
            target.health = min(target.health, 0)
            progress = True
            events.append(
                CombatEvent("poison", attacker.instance_id, target.instance_id)
            )
        if (
            not attacker_had_shield
            and MinionKeyword.POISONOUS in target.keywords
            and target.attack > 0
        ):
            attacker.health = min(attacker.health, 0)
            progress = True
            events.append(
                CombatEvent("poison", target.instance_id, attacker.instance_id)
            )
        _mark_health_deaths(boards, events)
        record("hit", attacker.instance_id, target.instance_id)
        if not progress and not any(
            minion.dead for board in boards for minion in board
        ):
            events.append(CombatEvent("stalemate", attacker.instance_id))
            break
        _resolve_deaths(
            boards,
            (attacker_side, defender_side),
            players,
            cards_by_id,
            bindings_by_id,
            events,
            rules,
            record,
        )
        active = defender_side

    winner_side = (
        0 if boards[0] and not boards[1] else 1 if boards[1] and not boards[0] else None
    )
    winner_id = players[winner_side].player_id if winner_side is not None else None
    loser_id = players[1 - winner_side].player_id if winner_side is not None else None
    damage = 0
    if winner_side is not None:
        winner = players[winner_side]
        damage = min(
            rules.combat_damage_cap,
            winner.tavern_tier
            + sum(
                1 if minion.token else max(1, cards_by_id[minion.definition_id].tier)
                for minion in boards[winner_side]
            ),
        )
        events.append(CombatEvent("hero_damage", winner.player_id, loser_id, damage))
    record("end")

    survivors = tuple(
        (
            players[side].player_id,
            tuple(
                CombatMinionResult(
                    instance_id=minion.instance_id,
                    definition_id=minion.definition_id,
                    attack=minion.attack,
                    health=minion.health,
                    max_health=minion.max_health,
                    keywords=tuple(minion.keywords),
                    token=minion.token,
                    token_name=minion.token_name,
                )
                for minion in boards[side]
                if not minion.dead
            ),
        )
        for side in range(2)
    )
    return CombatResult(
        winner_id=winner_id,
        loser_id=loser_id,
        damage=damage,
        attacks=attacks,
        first_attacker_id=first_attacker_id,
        survivors_by_player=survivors,
        events=tuple(events),
        frames=tuple(frames),
    )


def _snapshot(
    boards: list[list[_CombatMinion]],
) -> tuple[tuple[CombatMinionResult, ...], tuple[CombatMinionResult, ...]]:
    first, second = (
        tuple(
            CombatMinionResult(
                instance_id=minion.instance_id,
                definition_id=minion.definition_id,
                attack=minion.attack,
                health=minion.health,
                max_health=minion.max_health,
                keywords=tuple(minion.keywords),
                token=minion.token,
                token_name=minion.token_name,
            )
            for minion in board
        )
        for board in boards
    )
    return first, second


def _build_board(
    player: PlayerState,
    cards_by_id: dict[str, CardRecord],
    bindings_by_id: dict[str, CardEffectBinding],
) -> list[_CombatMinion]:
    board: list[_CombatMinion] = []
    for instance in player.board:
        if instance.token_spec is not None:
            board.append(
                _CombatMinion(
                    instance_id=instance.instance_id,
                    definition_id=instance.definition_id,
                    attack=instance.token_spec.attack + instance.attack_bonus,
                    health=instance.token_spec.health + instance.health_bonus,
                    max_health=instance.token_spec.health + instance.health_bonus,
                    keywords=list(
                        dict.fromkeys(
                            instance.token_spec.keywords + instance.granted_keywords
                        )
                    ),
                    token=True,
                    token_name=instance.token_spec.name,
                )
            )
            continue
        card = cards_by_id.get(instance.definition_id)
        if card is None:
            raise EffectResolutionError(
                f"Card definition {instance.definition_id!r} is missing from combat."
            )
        keywords = list(instance.granted_keywords)
        binding = bindings_by_id.get(card.card_id)
        if binding is not None:
            for effect in binding.effects:
                if (
                    effect.trigger is EffectTrigger.PASSIVE
                    and isinstance(effect.operation, KeywordEffect)
                    and effect.operation.target is EffectTarget.SELF
                    and effect.operation.keyword not in keywords
                ):
                    keywords.append(effect.operation.keyword)
        board.append(
            _CombatMinion(
                instance_id=instance.instance_id,
                definition_id=instance.definition_id,
                attack=card.attack + instance.attack_bonus,
                health=card.health + instance.health_bonus,
                max_health=card.health + instance.health_bonus,
                keywords=keywords,
                token=False,
            )
        )
    return board


def _next_alive_index(board: list[_CombatMinion], cursor: int) -> int | None:
    if not board:
        return None
    for offset in range(len(board)):
        index = (cursor + offset) % len(board)
        if not board[index].dead:
            return index
    return None


def _choose_target(
    board: list[_CombatMinion], rng: random.Random
) -> _CombatMinion | None:
    alive = [minion for minion in board if not minion.dead]
    if not alive:
        return None
    taunts = [minion for minion in alive if MinionKeyword.TAUNT in minion.keywords]
    return rng.choice(taunts or alive)


def _deal_attack_damage(
    source: _CombatMinion,
    target: _CombatMinion,
    events: list[CombatEvent],
) -> bool:
    amount = max(0, source.attack)
    if amount == 0:
        return False
    if MinionKeyword.DIVINE_SHIELD in target.keywords:
        target.keywords.remove(MinionKeyword.DIVINE_SHIELD)
        events.append(CombatEvent("shield", source.instance_id, target.instance_id))
        return True
    target.health -= amount
    return True


def _mark_health_deaths(
    boards: list[list[_CombatMinion]], events: list[CombatEvent]
) -> None:
    for board in boards:
        for minion in board:
            if minion.health <= 0 and not minion.dead:
                minion.dead = True
                events.append(CombatEvent("death", minion.instance_id))


def _resolve_deaths(
    boards: list[list[_CombatMinion]],
    resolution_order: tuple[int, int],
    players: tuple[PlayerState, PlayerState],
    cards_by_id: dict[str, CardRecord],
    bindings_by_id: dict[str, CardEffectBinding],
    events: list[CombatEvent],
    rules: GameRules,
    record: Callable[[str, str | None], None],
) -> None:
    dead_by_side = [[minion for minion in board if minion.dead] for board in boards]
    for board in boards:
        board[:] = [minion for minion in board if not minion.dead]
    if any(dead_by_side):
        record("deaths", None)

    for side in resolution_order:
        for dead in dead_by_side[side]:
            binding = bindings_by_id.get(dead.definition_id)
            if binding is None:
                continue
            for effect_index, effect in enumerate(binding.effects):
                if effect.trigger is not EffectTrigger.DEATHRATTLE:
                    continue
                if isinstance(effect.operation, SummonEffect):
                    _deathrattle_summon(
                        boards[side],
                        effect.operation,
                        dead,
                        effect_index,
                        rules,
                        events,
                    )
                elif isinstance(effect.operation, BuffEffect):
                    _deathrattle_buff(
                        boards[side],
                        effect.operation,
                        dead.instance_id,
                        cards_by_id,
                        bindings_by_id,
                        events,
                    )
                else:
                    raise EffectResolutionError(
                        "Unsupported combat Deathrattle operation "
                        f"{type(effect.operation).__name__}."
                    )
                record("deathrattle", dead.instance_id)
    for player, board in zip(players, boards, strict=True):
        if len(board) > rules.board_limit:
            raise EffectResolutionError(
                f"Combat board for {player.player_id!r} exceeded its configured limit."
            )


def _deathrattle_summon(
    board: list[_CombatMinion],
    effect: SummonEffect,
    dead: _CombatMinion,
    effect_index: int,
    rules: GameRules,
    events: list[CombatEvent],
) -> None:
    count = min(effect.count, max(0, rules.board_limit - len(board)))
    for index in range(count):
        token = _CombatMinion(
            instance_id=f"{dead.instance_id}:deathrattle:{effect_index}:{index}",
            definition_id=f"token:{dead.definition_id}:{effect_index}:{index}",
            attack=effect.token.attack,
            health=effect.token.health,
            max_health=effect.token.health,
            keywords=list(effect.token.keywords),
            token=True,
            token_name=effect.token.name,
        )
        board.append(token)
        events.append(CombatEvent("summon", dead.instance_id, token.instance_id))


def _deathrattle_buff(
    board: list[_CombatMinion],
    effect: BuffEffect,
    source_instance_id: str,
    cards_by_id: dict[str, CardRecord],
    bindings_by_id: dict[str, CardEffectBinding],
    events: list[CombatEvent],
) -> None:
    for minion in board:
        if _combat_target_matches(
            effect.target, minion, source_instance_id, cards_by_id, bindings_by_id
        ):
            minion.attack += effect.attack
            minion.health += effect.health
            minion.max_health += effect.health
            minion.keywords.extend(
                keyword for keyword in effect.keywords if keyword not in minion.keywords
            )
            events.append(CombatEvent("buff", source_instance_id, minion.instance_id))


def _combat_target_matches(
    target: EffectTarget,
    minion: _CombatMinion,
    source_instance_id: str,
    cards_by_id: dict[str, CardRecord],
    bindings_by_id: dict[str, CardEffectBinding],
) -> bool:
    if target is EffectTarget.ALL_FRIENDLY_MINIONS:
        return True
    if target is EffectTarget.SELF:
        return minion.instance_id == source_instance_id
    if target in {
        EffectTarget.OTHER_FRIENDLY_MECHS,
        EffectTarget.OTHER_FRIENDLY_MURLOCS,
    }:
        if minion.instance_id == source_instance_id:
            return False
    if target is EffectTarget.FRIENDLY_TAUNT_MINIONS:
        card = cards_by_id.get(minion.definition_id)
        binding = bindings_by_id.get(card.card_id) if card is not None else None
        has_profile_taunt = binding is not None and any(
            effect.trigger is EffectTrigger.PASSIVE
            and isinstance(effect.operation, KeywordEffect)
            and effect.operation.target is EffectTarget.SELF
            and effect.operation.keyword is MinionKeyword.TAUNT
            for effect in binding.effects
        )
        return MinionKeyword.TAUNT in minion.keywords or has_profile_taunt
    card = cards_by_id.get(minion.definition_id)
    if card is None:
        return False
    tribe_by_target = {
        EffectTarget.FRIENDLY_MURLOC: "Murloc",
        EffectTarget.OTHER_FRIENDLY_MURLOCS: "Murloc",
        EffectTarget.FRIENDLY_DEMON: "Demon",
        EffectTarget.FRIENDLY_DEMONS: "Demon",
        EffectTarget.FRIENDLY_MECH: "Mech",
        EffectTarget.OTHER_FRIENDLY_MECHS: "Mech",
        EffectTarget.FRIENDLY_BEAST: "Beast",
    }
    tribe = tribe_by_target.get(target)
    if tribe is None:
        raise EffectResolutionError(f"Unsupported combat target {target.value!r}.")
    return card.tribe == tribe or card.card_type == "All"

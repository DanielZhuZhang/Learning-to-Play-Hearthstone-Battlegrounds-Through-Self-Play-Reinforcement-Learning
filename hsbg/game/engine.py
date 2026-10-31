"""Headless match coordinator for recruit actions and seeded combat rounds."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass, replace

from hsbg.data.models import CardRecord, IterationProfile
from hsbg.game.actions import (
    InvalidActionError,
    buy_offer,
    reorder_board,
    reroll_shop,
    sell_minion,
    set_shop_frozen,
    start_recruit_turn,
    upgrade_tavern,
)
from hsbg.game.combat import CombatResult, resolve_combat
from hsbg.game.effects import play_minion_and_resolve_battlecry
from hsbg.game.legal import (
    RecruitAction,
    RecruitActionKind,
    get_legal_recruit_actions,
)
from hsbg.game.pool import MinionPool
from hsbg.game.setup import create_initial_match
from hsbg.game.state import MatchPhase, MatchState, PlayerState


@dataclass(frozen=True, slots=True)
class ActionRecord:
    """One accepted recruit action for deterministic replay."""

    player_id: str
    action: RecruitAction
    board_order: tuple[str, ...] | None = None


@dataclass(frozen=True, slots=True)
class MatchReplay:
    """Inputs and ordered player operations needed to reproduce a match."""

    iteration_id: str
    profile: IterationProfile
    selected_cards: tuple[CardRecord, ...]
    seed: int
    player_ids: tuple[str, ...]
    actions: tuple[ActionRecord, ...]


class GameEngine:
    """Mutable match coordinator built on immutable state snapshots."""

    def __init__(
        self,
        profile: IterationProfile,
        selected_cards: Sequence[CardRecord],
        seed: int,
        player_ids: Sequence[str] | None = None,
        *,
        record_combat_frames: bool = False,
    ) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("Match seed must be an integer.")
        self.profile = profile
        self.rules = profile.rules
        self.cards = tuple(selected_cards)
        if tuple(card.card_id for card in self.cards) != profile.enabled_card_ids:
            raise ValueError(
                "Selected cards must match the profile's enabled IDs and order."
            )
        self.seed = seed
        self.record_combat_frames = record_combat_frames
        self.rng = random.Random(seed)
        self._player_ids = tuple(
            player_ids
            if player_ids is not None
            else (f"player-{index + 1}" for index in range(self.rules.player_count))
        )
        slots = self.rules.shop_slots_by_tier[self.rules.starting_tavern_tier - 1]
        offer_groups = tuple(
            tuple(f"{player_id}:shop:{slot}" for slot in range(1, slots + 1))
            for player_id in self._player_ids
        )
        self.pool: MinionPool
        self.state, self.pool = create_initial_match(
            self.cards,
            self.rules,
            self._player_ids,
            offer_groups,
            self.rng,
        )
        self._ended_recruit: set[str] = set()
        self._next_offer_number = 0
        self._next_instance_number: dict[str, int] = {
            player_id: 0 for player_id in self._player_ids
        }
        self._recent_opponents: dict[str, list[str]] = {
            player_id: [] for player_id in self._player_ids
        }
        self._placements: dict[str, int] = {}
        self.action_history: list[ActionRecord] = []
        self.combat_history: list[CombatResult] = []

    @property
    def placements(self) -> tuple[tuple[str, int], ...]:
        """Known placements in the stable original player order."""
        return tuple(
            (player_id, self._placements[player_id])
            for player_id in self._player_ids
            if player_id in self._placements
        )

    @property
    def finished(self) -> bool:
        return self.state.phase is MatchPhase.FINISHED

    @property
    def pending_player_ids(self) -> tuple[str, ...]:
        """Living players that have not ended the current recruit phase."""
        if self.state.phase is not MatchPhase.RECRUIT:
            return ()
        return tuple(
            player.player_id
            for player in self.state.players
            if player.health > 0 and player.player_id not in self._ended_recruit
        )

    def legal_actions(self, player_id: str) -> tuple[RecruitAction, ...]:
        """Return legal actions for an active player in the recruit phase."""
        self._require_recruiting(player_id)
        if player_id in self._ended_recruit:
            raise InvalidActionError(f"Player {player_id!r} already ended recruitment.")
        player = self._player(player_id)
        if player.health <= 0:
            raise InvalidActionError(f"Player {player_id!r} has been eliminated.")
        return get_legal_recruit_actions(
            player, self.rules, self.cards, self.profile.card_effects
        )

    def player_state(self, player_id: str) -> PlayerState:
        """Return the current state snapshot for one configured player."""
        if player_id not in self._player_ids:
            raise InvalidActionError(f"Unknown player ID {player_id!r}.")
        return self._player(player_id)

    def apply_action(
        self,
        player_id: str,
        action: RecruitAction,
        *,
        board_order: Sequence[str] | None = None,
    ) -> None:
        """Validate and apply one structured recruit action."""
        if action not in self.legal_actions(player_id):
            raise InvalidActionError(f"Action {action!r} is not currently legal.")
        player = self._player(player_id)
        recorded_order: tuple[str, ...] | None = None
        kind = action.kind

        if kind is RecruitActionKind.BUY:
            self._next_instance_number[player_id] += 1
            instance_id = f"{player_id}:minion:{self._next_instance_number[player_id]}"
            updated = buy_offer(player, action.target_id or "", instance_id, self.rules)
            self._replace_player(updated)
        elif kind is RecruitActionKind.PLAY:
            updated = play_minion_and_resolve_battlecry(
                player,
                action.target_id or "",
                self.cards,
                self.profile.card_effects,
                self.rules,
                action.effect_target_id,
            )
            self._replace_player(updated)
        elif kind is RecruitActionKind.SELL:
            updated, self.pool = sell_minion(
                player, self.pool, action.target_id or "", self.rules
            )
            self._replace_player(updated)
        elif kind is RecruitActionKind.REROLL:
            offer_ids = self._new_offer_ids(player)
            eligible = self._eligible_card_ids(player.tavern_tier)
            updated, self.pool = reroll_shop(
                player, self.pool, eligible, offer_ids, self.rng, self.rules
            )
            self._next_offer_number += 1
            self._replace_player(updated)
        elif kind is RecruitActionKind.FREEZE:
            self._replace_player(set_shop_frozen(player, True))
        elif kind is RecruitActionKind.UNFREEZE:
            self._replace_player(set_shop_frozen(player, False))
        elif kind is RecruitActionKind.UPGRADE_TAVERN:
            self._replace_player(upgrade_tavern(player, self.rules))
        elif kind is RecruitActionKind.REORDER_BOARD:
            if board_order is None:
                raise InvalidActionError(
                    "Reordering the board requires an explicit order."
                )
            recorded_order = tuple(board_order)
            self._replace_player(reorder_board(player, recorded_order))
        elif kind is RecruitActionKind.END_RECRUIT:
            self._ended_recruit.add(player_id)
        else:
            raise InvalidActionError(f"Unsupported recruit action {kind.value!r}.")

        self.action_history.append(ActionRecord(player_id, action, recorded_order))
        if self._player(player_id).health <= 0:
            self._ended_recruit.add(player_id)
        if self._all_living_players_ended_recruit():
            self._resolve_round()

    def to_replay(self) -> MatchReplay:
        """Return an immutable deterministic replay description."""
        return MatchReplay(
            iteration_id=self.profile.iteration_id,
            profile=self.profile,
            selected_cards=self.cards,
            seed=self.seed,
            player_ids=self._player_ids,
            actions=tuple(self.action_history),
        )

    @classmethod
    def replay(
        cls,
        profile: IterationProfile,
        selected_cards: Sequence[CardRecord],
        replay: MatchReplay,
    ) -> GameEngine:
        """Recreate a match by reapplying its recorded actions and seed."""
        if replay.iteration_id != profile.iteration_id:
            raise ValueError(
                f"Replay iteration {replay.iteration_id!r} does not match profile "
                f"{profile.iteration_id!r}."
            )
        if replay.profile != profile:
            raise ValueError("Replay profile or effective rules do not match.")
        if replay.selected_cards != tuple(selected_cards):
            raise ValueError("Replay card data does not match the selected cards.")
        engine = cls(profile, selected_cards, replay.seed, replay.player_ids)
        for record in replay.actions:
            engine.apply_action(
                record.player_id,
                record.action,
                board_order=record.board_order,
            )
        return engine

    def _require_recruiting(self, player_id: str) -> None:
        if self.finished or self.state.phase is not MatchPhase.RECRUIT:
            raise InvalidActionError("The match is not in the recruit phase.")
        if player_id not in self._player_ids:
            raise InvalidActionError(f"Unknown player ID {player_id!r}.")

    def _player(self, player_id: str) -> PlayerState:
        return next(
            player for player in self.state.players if player.player_id == player_id
        )

    def _replace_player(self, updated: PlayerState) -> None:
        self.state = replace(
            self.state,
            players=tuple(
                updated if player.player_id == updated.player_id else player
                for player in self.state.players
            ),
        )

    def _all_living_players_ended_recruit(self) -> bool:
        living = {
            player.player_id for player in self.state.players if player.health > 0
        }
        return living.issubset(self._ended_recruit)

    def _eligible_card_ids(self, tavern_tier: int) -> tuple[str, ...]:
        return tuple(card.card_id for card in self.cards if card.tier <= tavern_tier)

    def _new_offer_ids(self, player: PlayerState) -> tuple[str, ...]:
        slots = self.rules.shop_slots_by_tier[player.tavern_tier - 1]
        number = self._next_offer_number + 1
        return tuple(
            f"{player.player_id}:shop:{number}:{slot}" for slot in range(1, slots + 1)
        )

    def _resolve_round(self) -> None:
        players_before = tuple(self.state.players)
        living = [
            player
            for player in players_before
            if player.health > 0 and player.player_id not in self._placements
        ]
        eliminated = [
            player
            for player in players_before
            if player.health <= 0 and player.player_id not in self._placements
        ]
        self._assign_elimination_placements(eliminated, len(living) + len(eliminated))

        if len(living) <= 1:
            if living:
                self._placements[living[0].player_id] = 1
            self.state = replace(self.state, phase=MatchPhase.FINISHED)
            return

        pairings, ghost_player_id = self._pair_players(living)
        results: list[CombatResult] = []
        updates: dict[str, PlayerState] = {
            player.player_id: player for player in living
        }
        for first_id, second_id in pairings:
            first = updates[first_id]
            second = updates[second_id]
            result = resolve_combat(
                first,
                second,
                self.cards,
                self.profile.card_effects,
                self.rng,
                self.rules,
                record_frames=self.record_combat_frames,
            )
            results.append(result)
            if result.loser_id is not None:
                loser = updates[result.loser_id]
                updates[loser.player_id] = _apply_hero_damage(loser, result.damage)

        if ghost_player_id is not None:
            ghost = PlayerState(
                player_id=f"ghost:{self.state.turn}",
                health=self.rules.starting_health,
                armor=0,
                gold=0,
                tavern_tier=self.rules.starting_tavern_tier,
            )
            result = resolve_combat(
                updates[ghost_player_id],
                ghost,
                self.cards,
                self.profile.card_effects,
                self.rng,
                self.rules,
                record_frames=self.record_combat_frames,
            )
            results.append(result)
            if result.loser_id == ghost_player_id:
                updates[ghost_player_id] = _apply_hero_damage(
                    updates[ghost_player_id], result.damage
                )

        self.combat_history.extend(results)
        self._record_pairings(pairings)
        self.state = replace(
            self.state,
            players=tuple(
                updates.get(player.player_id, player) for player in players_before
            ),
        )
        new_eliminated = [
            player
            for player in self.state.players
            if player.health <= 0 and player.player_id not in self._placements
        ]
        active = [
            player
            for player in self.state.players
            if player.health > 0 and player.player_id not in self._placements
        ]
        self._assign_elimination_placements(
            new_eliminated, len(active) + len(new_eliminated)
        )

        if len(active) <= 1:
            if active:
                self._placements[active[0].player_id] = 1
            self.state = replace(self.state, phase=MatchPhase.FINISHED)
            return
        if self.state.turn >= self.rules.max_match_turns:
            self._finish_by_remaining_health(active)
            return

        next_turn = self.state.turn + 1
        next_players: list[PlayerState] = []
        pool = self.pool
        for player in self.state.players:
            if player.health <= 0:
                next_players.append(player)
                continue
            offer_ids = self._new_offer_ids(player)
            updated_player, pool = start_recruit_turn(
                player,
                pool,
                next_turn,
                self._eligible_card_ids(player.tavern_tier),
                offer_ids,
                self.rng,
                self.rules,
            )
            self._next_offer_number += 1
            next_players.append(updated_player)
        self.pool = pool
        self._ended_recruit.clear()
        self.state = MatchState(
            turn=next_turn,
            phase=MatchPhase.RECRUIT,
            players=tuple(next_players),
        )

    def _pair_players(
        self, players: Sequence[PlayerState]
    ) -> tuple[list[tuple[str, str]], str | None]:
        player_ids = [player.player_id for player in players]
        ghost_id: str | None = None
        if len(player_ids) % 2:
            ghost_id = self.rng.choice(player_ids)
            player_ids.remove(ghost_id)

        all_pairings = _all_pairings(tuple(player_ids))
        if not all_pairings:
            return [], ghost_id
        recent_rounds = self.rules.rematch_avoidance_rounds
        scored = [
            (
                (
                    sum(
                        second in self._recent_opponents[first][-recent_rounds:]
                        for first, second in pairing
                    )
                    if recent_rounds
                    else 0
                ),
                pairing,
            )
            for pairing in all_pairings
        ]
        lowest_score = min(score for score, _ in scored)
        best_pairings = [pairing for score, pairing in scored if score == lowest_score]
        return list(self.rng.choice(best_pairings)), ghost_id

    def _record_pairings(self, pairings: Sequence[tuple[str, str]]) -> None:
        if not self.rules.rematch_avoidance_rounds:
            return
        for first, second in pairings:
            for player_id, opponent_id in ((first, second), (second, first)):
                recent = self._recent_opponents[player_id]
                recent.append(opponent_id)
                del recent[: -self.rules.rematch_avoidance_rounds]

    def _assign_elimination_placements(
        self,
        eliminated: Sequence[PlayerState],
        unplaced_count: int,
    ) -> None:
        if not eliminated:
            return
        place = unplaced_count - len(eliminated) + 1
        for player in eliminated:
            self._placements[player.player_id] = place

    def _finish_by_remaining_health(self, active: Sequence[PlayerState]) -> None:
        ordered = sorted(
            active,
            key=lambda player: (
                player.health + player.armor,
                player.health,
                player.armor,
            ),
            reverse=True,
        )
        index = 0
        while index < len(ordered):
            end = index + 1
            score = (
                ordered[index].health + ordered[index].armor,
                ordered[index].health,
                ordered[index].armor,
            )
            while (
                end < len(ordered)
                and (
                    ordered[end].health + ordered[end].armor,
                    ordered[end].health,
                    ordered[end].armor,
                )
                == score
            ):
                end += 1
            place = index + 1
            for player in ordered[index:end]:
                self._placements[player.player_id] = place
            index = end
        self.state = replace(self.state, phase=MatchPhase.FINISHED)


def _all_pairings(player_ids: tuple[str, ...]) -> list[tuple[tuple[str, str], ...]]:
    if not player_ids:
        return [()]
    first = player_ids[0]
    pairings: list[tuple[tuple[str, str], ...]] = []
    for opponent in player_ids[1:]:
        remaining = tuple(
            player_id for player_id in player_ids[1:] if player_id != opponent
        )
        pairings.extend(
            ((first, opponent),) + tail for tail in _all_pairings(remaining)
        )
    return pairings


def _apply_hero_damage(player: PlayerState, amount: int) -> PlayerState:
    armor_damage = min(player.armor, amount)
    health_damage = amount - armor_damage
    return replace(
        player,
        armor=player.armor - armor_damage,
        health=max(0, player.health - health_damage),
    )

"""Pygame client that draws the match like a Battlegrounds table.

Every rule still runs through ``GameEngine``; this module only chooses bot
actions, forwards clicks, and animates the recorded combat frames.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pygame

from hsbg.data.models import (
    CardRecord,
    EffectTarget,
    EffectTrigger,
    IterationProfile,
    KeywordEffect,
    MinionKeyword,
)
from hsbg.game.combat import CombatFrame, CombatMinionResult, CombatResult
from hsbg.game.engine import ActionRecord, GameEngine, MatchReplay
from hsbg.game.legal import RecruitAction, RecruitActionKind
from hsbg.game.state import MatchState, MinionInstance, PlayerState, ShopOffer
from hsbg.ui_common import ALL_PLAYERS, EventLine, describe_action, visible_lines

if TYPE_CHECKING:
    from hsbg.agent_bot import TrainedBot

WIDTH, HEIGHT = 1280, 760
FPS = 60
SPEEDS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
BOT_ACTION_DELAY = 550
MAX_BOT_ACTIONS_PER_TURN = 32
FRAME_DURATIONS = {
    "start": 1100,
    "attack": 420,
    "hit": 650,
    "deaths": 380,
    "deathrattle": 650,
    "end": 2000,
}

# Layout: leaderboard | table | log, with a top status bar and bottom controls.
LEFT_X, LEFT_W = 8, 200
MAIN_X, MAIN_W = 216, 784
LOG_X, LOG_W = 1008, 264
TOP_H = 52
BOTTOM_Y = 712
CARD_W, CARD_H = 104, 148
MINION_W, MINION_H = 100, 128
SHOP_Y, BOARD_Y, HAND_Y = 80, 264, 458

BG = (22, 17, 13)
TABLE = (56, 41, 29)
TABLE_EDGE = (98, 74, 50)
PANEL = (34, 27, 21)
TEXT = (240, 230, 212)
MUTED = (165, 150, 128)
GOLD = (242, 196, 64)
RED = (214, 58, 52)
GREEN = (110, 214, 104)
FROST = (130, 195, 255)
CARD_BODY = (82, 63, 46)
CARD_EDGE = (160, 128, 84)
BUTTON = (92, 66, 40)
BUTTON_HOVER = (128, 92, 52)
BUTTON_OFF = (60, 52, 46)
TRIBE_COLORS = {
    "Beast": (118, 150, 70),
    "Demon": (128, 72, 150),
    "Mech": (106, 124, 142),
    "Murloc": (58, 140, 138),
    "All": (190, 150, 60),
}
NEUTRAL_TRIBE = (138, 110, 88)
HERO_COLORS = (
    (176, 72, 64),
    (64, 120, 176),
    (88, 150, 80),
    (170, 128, 52),
    (128, 84, 160),
    (60, 150, 150),
    (176, 96, 140),
    (120, 120, 120),
)


@dataclass(frozen=True, slots=True)
class MinionView:
    """Everything needed to draw one card or minion, from any source."""

    key: str
    name: str
    attack: int
    health: int
    base_attack: int
    base_health: int
    max_health: int
    tier: int
    tribe: str | None
    text: str
    keywords: frozenset[MinionKeyword]
    deathrattle: bool


@dataclass(slots=True)
class FloatText:
    text: str
    color: tuple[int, int, int]
    x: float
    y: float
    start: float
    duration: float = 1100.0


@dataclass(slots=True)
class CombatView:
    """Playback position within one recorded combat."""

    result: CombatResult
    bottom: int
    heroes_before: tuple[PlayerState | None, PlayerState | None]
    heroes_after: tuple[PlayerState | None, PlayerState | None]
    turn: int
    players_before: tuple[PlayerState, ...]
    log_line: EventLine
    frame: int = 0
    frame_start: float | None = None


class PygameMatchClient:
    """Battlegrounds-style table view over a ``GameEngine`` match."""

    def __init__(
        self,
        profile: IterationProfile,
        selected_cards: Sequence[CardRecord],
        seed: int,
        human_player_id: str | None,
        pov_player_id: str | None = None,
        agents: Mapping[str, TrainedBot] | None = None,
    ) -> None:
        self.agents = dict(agents or {})
        self.profile = profile
        self.rules = profile.rules
        self.cards = tuple(selected_cards)
        self.card_by_id = {card.card_id: card for card in self.cards}
        self.bindings = {binding.card_id: binding for binding in profile.card_effects}
        self.seed = seed
        self.human_player_id = human_player_id
        self.engine = GameEngine(profile, self.cards, seed, record_combat_frames=True)
        self.player_ids = tuple(player.player_id for player in self.engine.state.players)
        self.view_options = (ALL_PLAYERS, *self.player_ids)
        initial_view = pov_player_id or human_player_id
        self.view_index = (
            self.view_options.index(initial_view)
            if initial_view in self.view_options
            else 0
        )
        self.bot_rng = random.Random(seed ^ 0xB075)
        self._bot_turn = self.engine.state.turn
        self._bot_counts: dict[str, int] = {}

        self.log: list[EventLine] = [
            (
                "Welcome to the tavern! Buy minions, play them onto your board, "
                "then End Turn. Combat plays out automatically.",
                None,
            )
        ]
        self.log_scroll = 0
        self.floats: list[FloatText] = []
        self.combats: list[CombatView] = []
        self.selected_id: str | None = None
        # Hand minion waiting for the player to pick its Battlecry target.
        self.targeting_id: str | None = None
        self.speed_index = SPEEDS.index(1.0)
        self.paused = False
        self.step_requested = False
        self.clock_ms = 0.0
        self.next_action_at = 0.0
        self.last_replay: MatchReplay | None = None
        self.replay_actions: Sequence[ActionRecord] = ()
        self.replay_index = 0
        self.replaying = False
        self.running = True
        self.hits: list[tuple[pygame.Rect, Callable[[], None]]] = []
        self.hovered: MinionView | None = None

        pygame.init()
        pygame.display.set_caption("Battlegrounds Simulator")
        try:
            self.screen = pygame.display.set_mode(
                (WIDTH, HEIGHT), pygame.SCALED | pygame.RESIZABLE
            )
        except pygame.error:
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.timer = pygame.time.Clock()
        self._fonts: dict[tuple[int, bool], pygame.font.Font] = {}

    # ------------------------------------------------------------------ loop

    def run(self) -> None:
        """Run until the window is closed."""
        while self.running:
            elapsed = self.timer.tick(FPS)
            self.handle_events(pygame.event.get())
            self.update(elapsed)
            self.draw()
            pygame.display.flip()
        pygame.quit()

    def handle_events(self, events: Sequence[pygame.event.Event]) -> None:
        for event in events:
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for rect, callback in reversed(self.hits):
                    if rect.collidepoint(event.pos):
                        callback()
                        break
                else:
                    self.selected_id = None
                    self.targeting_id = None
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
                self.targeting_id = None
            elif event.type == pygame.MOUSEWHEEL:
                if pygame.Rect(LOG_X, TOP_H, LOG_W, BOTTOM_Y - TOP_H).collidepoint(
                    pygame.mouse.get_pos()
                ):
                    self.log_scroll = max(0, self.log_scroll + event.y * 3)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self._toggle_pause()
                elif event.key == pygame.K_n:
                    self._request_step()
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    self._change_speed(1)
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    self._change_speed(-1)
                elif event.key == pygame.K_ESCAPE:
                    if self.targeting_id is not None:
                        self.targeting_id = None
                    else:
                        self._skip_combats()

    def update(self, elapsed_ms: float) -> None:
        """Advance animations and let bots (or the replay) act."""
        if not self.paused:
            self.clock_ms += elapsed_ms * SPEEDS[self.speed_index]
        self.floats = [
            text for text in self.floats if self.clock_ms - text.start < text.duration
        ]
        if self.combats:
            self._update_combat()
            return
        if self.engine.finished:
            return
        forced = self.paused and self.step_requested
        if self.paused and not forced:
            return
        if not forced and self.clock_ms < self.next_action_at:
            return
        # Hidden actions (other players, when viewing one player) run instantly.
        for _ in range(500):
            actor = self._automatic_actor()
            if actor is None:
                break
            visible = self.pov is None or actor == self.pov
            self._take_automatic_action(actor)
            if visible or self.combats or self.engine.finished:
                self.next_action_at = self.clock_ms + BOT_ACTION_DELAY
                break
        self.step_requested = False

    # --------------------------------------------------------------- engine

    @property
    def pov(self) -> str | None:
        choice = self.view_options[self.view_index]
        return None if choice == ALL_PLAYERS else choice

    def _seat_tag(self, player_id: str) -> str:
        if player_id == self.human_player_id:
            return " (you)"
        if player_id in self.agents:
            return " (agent)"
        return ""

    def _automatic_actor(self) -> str | None:
        pending = self.engine.pending_player_ids
        if not pending or self.engine.finished:
            return None
        if self.replaying:
            if self.replay_index >= len(self.replay_actions):
                self.replaying = False
                return None
            return self.replay_actions[self.replay_index].player_id
        if pending[0] == self.human_player_id:
            return None
        return pending[0]

    def _take_automatic_action(self, player_id: str) -> None:
        if self.replaying:
            record = self.replay_actions[self.replay_index]
            self.replay_index += 1
            self._apply(record.player_id, record.action, record.board_order)
            return
        if self.engine.state.turn != self._bot_turn:
            self._bot_turn = self.engine.state.turn
            self._bot_counts.clear()
        agent = self.agents.get(player_id)
        if agent is not None:
            count = self._bot_counts.get(player_id, 0)
            if count >= agent.max_actions_per_turn:
                self._apply(player_id, RecruitAction(RecruitActionKind.END_RECRUIT))
                return
            move = agent.choose_move(self.engine, player_id)
            if move.action.kind is not RecruitActionKind.END_RECRUIT:
                self._bot_counts[player_id] = count + 1
            self._apply(player_id, move.action, move.board_order)
            return
        legal = self.engine.legal_actions(player_id)
        if self._bot_counts.get(player_id, 0) >= MAX_BOT_ACTIONS_PER_TURN:
            action = next(a for a in legal if a.kind is RecruitActionKind.END_RECRUIT)
        else:
            action = self.bot_rng.choice(legal)
        board_order = None
        if action.kind is RecruitActionKind.REORDER_BOARD:
            board = self.engine.player_state(player_id).board
            board_order = tuple(
                minion.instance_id
                for minion in self.bot_rng.sample(list(board), len(board))
            )
        if action.kind is not RecruitActionKind.END_RECRUIT:
            self._bot_counts[player_id] = self._bot_counts.get(player_id, 0) + 1
        self._apply(player_id, action, board_order)

    def _apply(
        self,
        player_id: str,
        action: RecruitAction,
        board_order: tuple[str, ...] | None = None,
    ) -> None:
        state_before = self.engine.state
        player_before = self.engine.player_state(player_id)
        combats_before = len(self.engine.combat_history)
        self.engine.apply_action(player_id, action, board_order=board_order)
        record = self.engine.action_history[-1]
        self.log.append(
            (
                describe_action(
                    record,
                    player_before,
                    self.rules,
                    self.card_by_id,
                    self.human_player_id,
                ),
                frozenset({player_id}),
            )
        )
        if action.kind is RecruitActionKind.PLAY:
            self._float_recruit_changes(player_before)
        for result in self.engine.combat_history[combats_before:]:
            line = self._combat_log_line(result)
            if self.pov is None or self.pov in _combatants(result):
                # Revealed when the animation finishes, so the log doesn't spoil it.
                self.combats.append(self._combat_view(result, state_before, line))
            else:
                self.log.append(line)
        self.targeting_id = None
        if self.combats:
            self.selected_id = None
        if self.engine.finished and not self.replaying:
            self.last_replay = self.engine.to_replay()

    def _combat_log_line(self, result: CombatResult) -> EventLine:
        first, second = _combatants(result)
        if result.winner_id is None:
            text = f"Combat: {_hero_label(first)} vs {_hero_label(second)} tied."
        else:
            text = (
                f"Combat: {_hero_label(result.winner_id)} beat "
                f"{_hero_label(result.loser_id or '')}, dealing {result.damage} damage."
            )
        return text, frozenset((first, second))

    def _combat_view(
        self, result: CombatResult, state_before: MatchState, log_line: EventLine
    ) -> CombatView:
        ids = _combatants(result)
        focus = self.pov or self.human_player_id
        bottom = ids.index(focus) if focus in ids else 0
        before = {player.player_id: player for player in state_before.players}
        return CombatView(
            result=result,
            bottom=bottom,
            heroes_before=(before.get(ids[0]), before.get(ids[1])),
            heroes_after=tuple(
                self.engine.player_state(pid) if pid in self.player_ids else None
                for pid in ids
            ),
            turn=state_before.turn,
            players_before=state_before.players,
            log_line=log_line,
        )

    def _human_action(self, action: RecruitAction, order: tuple[str, ...] | None = None) -> None:
        if not self._can_act():
            return
        if action not in self.engine.legal_actions(self.human_player_id or ""):
            return
        self._apply(self.human_player_id or "", action, order)
        self.next_action_at = self.clock_ms + BOT_ACTION_DELAY

    def _can_act(self) -> bool:
        pending = self.engine.pending_player_ids
        return (
            not self.replaying
            and not self.combats
            and self.human_player_id is not None
            and bool(pending)
            and pending[0] == self.human_player_id
            and self._displayed_player_id() == self.human_player_id
        )

    # ------------------------------------------------------------- controls

    def _toggle_pause(self) -> None:
        self.paused = not self.paused

    def _request_step(self) -> None:
        if not self.paused:
            return
        if self.combats:
            self._advance_combat_frame(self.combats[0])
        else:
            self.step_requested = True

    def _change_speed(self, delta: int) -> None:
        self.speed_index = max(0, min(len(SPEEDS) - 1, self.speed_index + delta))

    def _cycle_view(self, delta: int) -> None:
        self.view_index = (self.view_index + delta) % len(self.view_options)
        self.selected_id = None
        self.log_scroll = 0

    def _set_view(self, player_id: str) -> None:
        self.view_index = self.view_options.index(player_id)
        self.selected_id = None
        self.log_scroll = 0

    def _skip_combats(self) -> None:
        while self.combats:
            self._finish_combat()

    def _finish_combat(self) -> None:
        self.log.append(self.combats.pop(0).log_line)
        self.log_scroll = 0
        self.next_action_at = self.clock_ms + BOT_ACTION_DELAY

    def _start_replay(self) -> None:
        if self.last_replay is None:
            return
        replay = self.last_replay
        self.engine = GameEngine(
            self.profile,
            self.cards,
            replay.seed,
            replay.player_ids,
            record_combat_frames=True,
        )
        self.replay_actions = replay.actions
        self.replay_index = 0
        self.replaying = True
        self.combats.clear()
        self.floats.clear()
        self.selected_id = None
        self.paused = False
        self.log = [("Replaying the recorded match from its first action.", None)]
        self.log_scroll = 0
        self.next_action_at = self.clock_ms

    # --------------------------------------------------------------- combat

    def _update_combat(self) -> None:
        view = self.combats[0]
        if not view.result.frames:
            self._finish_combat()
            return
        if view.frame_start is None:
            view.frame_start = self.clock_ms
            self._enter_frame(view)
        frame = view.result.frames[view.frame]
        if self.clock_ms - view.frame_start >= FRAME_DURATIONS.get(frame.kind, 500):
            self._advance_combat_frame(view)

    def _advance_combat_frame(self, view: CombatView) -> None:
        view.frame += 1
        if view.frame >= len(view.result.frames):
            self._finish_combat()
            return
        view.frame_start = self.clock_ms
        self._enter_frame(view)

    def _enter_frame(self, view: CombatView) -> None:
        frames = view.result.frames
        frame = frames[view.frame]
        rects = self._combat_rects(view, frame)
        if frame.kind == "end":
            loser = view.result.loser_id
            if loser is not None and view.result.damage:
                side = _combatants(view.result).index(loser)
                x, y = self._hero_center(view, side)
                self._float(f"-{view.result.damage}", RED, x, y - 10, 1800)
            return
        if view.frame == 0:
            return
        previous = {
            minion.instance_id: minion
            for board in frames[view.frame - 1].boards
            for minion in board
        }
        for board in frame.boards:
            for minion in board:
                rect = rects.get(minion.instance_id)
                if rect is None:
                    continue
                old = previous.get(minion.instance_id)
                if old is None:
                    self._float("Summoned!", GREEN, rect.centerx, rect.top + 20)
                    continue
                health_change = minion.health - old.health
                attack_change = minion.attack - old.attack
                if frame.kind == "hit":
                    if health_change < 0:
                        self._float(f"-{-health_change}", RED, rect.centerx, rect.centery)
                    if (
                        MinionKeyword.DIVINE_SHIELD in old.keywords
                        and MinionKeyword.DIVINE_SHIELD not in minion.keywords
                    ):
                        self._float("Shield popped", GOLD, rect.centerx, rect.top + 10)
                elif attack_change > 0 or health_change > 0:
                    self._float(
                        f"+{attack_change}/+{health_change}",
                        GREEN,
                        rect.centerx,
                        rect.centery,
                    )

    def _float(
        self,
        text: str,
        color: tuple[int, int, int],
        x: float,
        y: float,
        duration: float = 1100.0,
    ) -> None:
        self.floats.append(FloatText(text, color, x, y, self.clock_ms, duration))

    def _float_recruit_changes(self, before: PlayerState) -> None:
        after = self.engine.player_state(before.player_id)
        if self._displayed_player_id() != after.player_id:
            return
        old = {minion.instance_id: minion for minion in before.minions}
        rects = _row_rects(len(after.board), BOARD_Y, MINION_W, MINION_H, gap=12)
        for rect, minion in zip(rects, after.board):
            previous = old.get(minion.instance_id)
            if previous is None:
                if minion.token_spec is not None:
                    self._float("Summoned!", GREEN, rect.centerx, rect.top + 20)
                continue
            attack = minion.attack_bonus - previous.attack_bonus
            health = minion.health_bonus - previous.health_bonus
            if attack or health:
                self._float(f"+{attack}/+{health}", GREEN, rect.centerx, rect.centery)
        if after.health < before.health:
            self._float(
                f"Hero -{before.health - after.health}",
                RED,
                MAIN_X + MAIN_W // 2,
                HAND_Y - 20,
            )

    # ---------------------------------------------------------------- views

    def _displayed_player_id(self) -> str:
        if self.pov is not None:
            return self.pov
        pending = self.engine.pending_player_ids
        if pending:
            return pending[0]
        return self.human_player_id or self.player_ids[0]

    def _keywords_for(self, definition_id: str) -> set[MinionKeyword]:
        binding = self.bindings.get(definition_id)
        if binding is None:
            return set()
        return {
            effect.operation.keyword
            for effect in binding.effects
            if effect.trigger is EffectTrigger.PASSIVE
            and isinstance(effect.operation, KeywordEffect)
            and effect.operation.target is EffectTarget.SELF
        }

    def _has_deathrattle(self, definition_id: str) -> bool:
        binding = self.bindings.get(definition_id)
        return binding is not None and any(
            effect.trigger is EffectTrigger.DEATHRATTLE for effect in binding.effects
        )

    def _card_view(
        self,
        key: str,
        definition_id: str,
        attack_bonus: int,
        health_bonus: int,
        granted: Sequence[MinionKeyword],
    ) -> MinionView:
        card = self.card_by_id[definition_id]
        attack = card.attack + attack_bonus
        health = card.health + health_bonus
        return MinionView(
            key=key,
            name=card.name,
            attack=attack,
            health=health,
            base_attack=card.attack,
            base_health=card.health,
            max_health=health,
            tier=card.tier,
            tribe=_tribe(card),
            text=" ".join(card.text.split()),
            keywords=frozenset(self._keywords_for(definition_id) | set(granted)),
            deathrattle=self._has_deathrattle(definition_id),
        )

    def _offer_view(self, offer: ShopOffer) -> MinionView:
        return self._card_view(
            offer.offer_id,
            offer.definition_id,
            offer.attack_bonus,
            offer.health_bonus,
            offer.granted_keywords,
        )

    def _instance_view(self, minion: MinionInstance) -> MinionView:
        if minion.token_spec is None:
            return self._card_view(
                minion.instance_id,
                minion.definition_id,
                minion.attack_bonus,
                minion.health_bonus,
                minion.granted_keywords,
            )
        token = minion.token_spec
        attack = token.attack + minion.attack_bonus
        health = token.health + minion.health_bonus
        return MinionView(
            key=minion.instance_id,
            name=token.name,
            attack=attack,
            health=health,
            base_attack=token.attack,
            base_health=token.health,
            max_health=health,
            tier=1,
            tribe=None,
            text="Summoned token.",
            keywords=frozenset(token.keywords + minion.granted_keywords),
            deathrattle=False,
        )

    def _combat_minion_view(self, minion: CombatMinionResult) -> MinionView:
        card = None if minion.token else self.card_by_id.get(minion.definition_id)
        return MinionView(
            key=minion.instance_id,
            name=card.name if card else (minion.token_name or "Token"),
            attack=minion.attack,
            health=minion.health,
            base_attack=card.attack if card else minion.attack,
            base_health=card.health if card else minion.max_health,
            max_health=minion.max_health,
            tier=card.tier if card else 1,
            tribe=_tribe(card) if card else None,
            text=" ".join(card.text.split()) if card else "Summoned token.",
            keywords=frozenset(minion.keywords),
            deathrattle=card is not None and self._has_deathrattle(card.card_id),
        )

    # -------------------------------------------------------------- drawing

    def font(self, size: int, bold: bool = False) -> pygame.font.Font:
        key = (size, bold)
        if key not in self._fonts:
            self._fonts[key] = pygame.font.SysFont("segoeui,arial", size, bold=bold)
        return self._fonts[key]

    def draw(self) -> None:
        self.hits = []
        self.hovered = None
        self.screen.fill(BG)
        pygame.draw.rect(
            self.screen,
            TABLE,
            (MAIN_X - 4, TOP_H, MAIN_W + 8, BOTTOM_Y - TOP_H - 6),
            border_radius=14,
        )
        pygame.draw.rect(
            self.screen,
            TABLE_EDGE,
            (MAIN_X - 4, TOP_H, MAIN_W + 8, BOTTOM_Y - TOP_H - 6),
            2,
            border_radius=14,
        )
        self._draw_leaderboard()
        self._draw_log()
        if self.combats and self.combats[0].result.frames:
            self._draw_combat(self.combats[0])
        else:
            self._draw_recruit()
        self._draw_bottom_bar()
        self._draw_floats()
        if self.engine.finished and not self.combats:
            self._draw_game_over()
        if self.hovered is not None:
            self._draw_tooltip(self.hovered)

    def _draw_text(
        self,
        text: str,
        size: int,
        color: tuple[int, int, int],
        pos: tuple[float, float],
        *,
        bold: bool = False,
        anchor: str = "topleft",
        outline: bool = False,
    ) -> pygame.Rect:
        font = self.font(size, bold)
        image = font.render(text, True, color)
        rect = image.get_rect(**{anchor: (round(pos[0]), round(pos[1]))})
        if outline:
            shadow = font.render(text, True, (12, 8, 6))
            for dx, dy in ((-2, 0), (2, 0), (0, -2), (0, 2), (-1, -1), (1, 1), (-1, 1), (1, -1)):
                self.screen.blit(shadow, rect.move(dx, dy))
        self.screen.blit(image, rect)
        return rect

    def _fit_size(self, text: str, max_width: int, size: int, bold: bool = True) -> int:
        while size > 9 and self.font(size, bold).size(text)[0] > max_width:
            size -= 1
        return size

    def _wrap(self, text: str, size: int, width: int) -> list[str]:
        font = self.font(size)
        lines: list[str] = []
        current = ""
        for word in text.split():
            candidate = f"{current} {word}".strip()
            if font.size(candidate)[0] <= width or not current:
                current = candidate
            else:
                lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines

    def _button(
        self,
        rect: pygame.Rect,
        label: str,
        callback: Callable[[], None] | None,
        *,
        size: int = 15,
        accent: tuple[int, int, int] | None = None,
    ) -> None:
        enabled = callback is not None
        hovered = enabled and rect.collidepoint(pygame.mouse.get_pos())
        fill = accent if accent and enabled else BUTTON
        if hovered:
            fill = tuple(min(255, channel + 36) for channel in fill)
        if not enabled:
            fill = BUTTON_OFF
        pygame.draw.rect(self.screen, fill, rect, border_radius=8)
        pygame.draw.rect(self.screen, CARD_EDGE if enabled else (80, 72, 64), rect, 2, border_radius=8)
        self._draw_text(
            label,
            self._fit_size(label, rect.width - 10, size),
            TEXT if enabled else MUTED,
            rect.center,
            bold=True,
            anchor="center",
        )
        if callback is not None:
            self.hits.append((rect, callback))

    def _draw_star(self, center: tuple[float, float], radius: float, color: tuple[int, int, int]) -> None:
        points = []
        for index in range(10):
            angle = -math.pi / 2 + index * math.pi / 5
            length = radius if index % 2 == 0 else radius * 0.45
            points.append((center[0] + math.cos(angle) * length, center[1] + math.sin(angle) * length))
        pygame.draw.polygon(self.screen, color, points)
        pygame.draw.polygon(self.screen, (60, 40, 10), points, 1)

    def _draw_tier(self, center_x: float, y: float, tier: int, radius: float = 6) -> None:
        spacing = radius * 2.1
        start = center_x - spacing * (tier - 1) / 2
        for index in range(tier):
            self._draw_star((start + index * spacing, y), radius, GOLD)

    def _draw_gem(
        self,
        center: tuple[float, float],
        value: int,
        fill: tuple[int, int, int],
        text_color: tuple[int, int, int],
        radius: int = 15,
    ) -> None:
        pygame.draw.circle(self.screen, (20, 14, 10), center, radius + 2)
        pygame.draw.circle(self.screen, fill, center, radius)
        self._draw_text(str(value), 18, text_color, center, bold=True, anchor="center", outline=True)

    def _stat_colors(self, view: MinionView) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        attack = GREEN if view.attack > view.base_attack else TEXT
        if view.health < view.max_health:
            health = (255, 120, 110)
        elif view.health > view.base_health:
            health = GREEN
        else:
            health = TEXT
        return attack, health

    def _draw_card(
        self,
        rect: pygame.Rect,
        view: MinionView,
        *,
        frozen: bool = False,
        clickable: Callable[[], None] | None = None,
    ) -> None:
        hovered = rect.collidepoint(pygame.mouse.get_pos())
        if hovered:
            self.hovered = view
        pygame.draw.rect(self.screen, CARD_BODY, rect, border_radius=10)
        edge = GOLD if hovered and clickable else CARD_EDGE
        pygame.draw.rect(self.screen, edge, rect, 3 if hovered else 2, border_radius=10)
        art = pygame.Rect(rect.x + 12, rect.y + 16, rect.width - 24, 58)
        pygame.draw.ellipse(self.screen, TRIBE_COLORS.get(view.tribe or "", NEUTRAL_TRIBE), art)
        pygame.draw.ellipse(self.screen, (40, 30, 20), art, 2)
        self._draw_tier(rect.centerx, rect.y + 10, view.tier, 5)
        banner = pygame.Rect(rect.x + 4, art.bottom + 2, rect.width - 8, 20)
        pygame.draw.rect(self.screen, (38, 28, 20), banner, border_radius=4)
        self._draw_text(
            view.name,
            self._fit_size(view.name, banner.width - 6, 13),
            TEXT,
            banner.center,
            bold=True,
            anchor="center",
        )
        y = banner.bottom + 2
        for line in self._wrap(view.text, 10, rect.width - 12)[:3]:
            self._draw_text(line, 10, MUTED, (rect.centerx, y), anchor="midtop")
            y += 12
        if view.tribe:
            self._draw_text(view.tribe, 10, MUTED, (rect.centerx, rect.bottom - 14), anchor="center")
        attack_color, health_color = self._stat_colors(view)
        self._draw_gem((rect.x + 14, rect.bottom - 14), view.attack, (204, 152, 40), attack_color, 14)
        self._draw_gem((rect.right - 14, rect.bottom - 14), view.health, (176, 40, 36), health_color, 14)
        if frozen:
            overlay = pygame.Surface(rect.size, pygame.SRCALPHA)
            pygame.draw.rect(overlay, (*FROST, 80), overlay.get_rect(), border_radius=10)
            pygame.draw.rect(overlay, (*FROST, 220), overlay.get_rect(), 3, border_radius=10)
            self.screen.blit(overlay, rect)
        if clickable is not None:
            self.hits.append((rect, clickable))

    def _draw_minion(
        self,
        rect: pygame.Rect,
        view: MinionView,
        *,
        selected: bool = False,
        targetable: bool = False,
        clickable: Callable[[], None] | None = None,
    ) -> None:
        if rect.collidepoint(pygame.mouse.get_pos()):
            self.hovered = view
        dead = view.health <= 0
        if MinionKeyword.TAUNT in view.keywords:
            shield = rect.inflate(14, 6)
            pygame.draw.rect(self.screen, (128, 128, 136), shield, border_radius=22)
            pygame.draw.rect(self.screen, (70, 70, 78), shield, 3, border_radius=22)
        if targetable:
            pygame.draw.ellipse(self.screen, GREEN, rect.inflate(14, 14), 5)
        if selected:
            pygame.draw.ellipse(self.screen, GOLD, rect.inflate(10, 10), 4)
        pygame.draw.ellipse(self.screen, (40, 30, 20), rect.inflate(4, 4))
        pygame.draw.ellipse(self.screen, TRIBE_COLORS.get(view.tribe or "", NEUTRAL_TRIBE), rect)
        pygame.draw.ellipse(self.screen, (214, 176, 98), rect, 3)
        self._draw_tier(rect.centerx, rect.y + 14, view.tier, 5)
        plaque = pygame.Rect(rect.x + 2, rect.centery - 2, rect.width - 4, 22)
        pygame.draw.rect(self.screen, (30, 22, 16), plaque, border_radius=6)
        self._draw_text(
            view.name,
            self._fit_size(view.name, plaque.width - 6, 13),
            TEXT,
            plaque.center,
            bold=True,
            anchor="center",
        )
        if view.deathrattle:
            center = (rect.centerx, rect.bottom - 10)
            pygame.draw.circle(self.screen, (52, 36, 70), center, 11)
            pygame.draw.circle(self.screen, (170, 140, 200), center, 11, 1)
            self._draw_text("DR", 10, (220, 200, 240), center, bold=True, anchor="center")
        if MinionKeyword.POISONOUS in view.keywords:
            center = (rect.right - 10, rect.y + 22)
            pygame.draw.circle(self.screen, (40, 150, 50), center, 10)
            self._draw_text("P", 12, TEXT, center, bold=True, anchor="center")
        if MinionKeyword.DIVINE_SHIELD in view.keywords:
            glow = pygame.Surface(rect.inflate(16, 16).size, pygame.SRCALPHA)
            pygame.draw.ellipse(glow, (255, 220, 100, 70), glow.get_rect())
            pygame.draw.ellipse(glow, (255, 232, 130, 230), glow.get_rect(), 3)
            self.screen.blit(glow, rect.inflate(16, 16))
        attack_color, health_color = self._stat_colors(view)
        self._draw_gem((rect.x + 8, rect.bottom - 16), view.attack, (204, 152, 40), attack_color)
        self._draw_gem((rect.right - 8, rect.bottom - 16), view.health, (176, 40, 36), health_color)
        if dead:
            shade = pygame.Surface(rect.size, pygame.SRCALPHA)
            pygame.draw.ellipse(shade, (10, 10, 10, 150), shade.get_rect())
            self.screen.blit(shade, rect)
            pygame.draw.line(self.screen, RED, rect.inflate(-30, -40).topleft, rect.inflate(-30, -40).bottomright, 6)
            pygame.draw.line(self.screen, RED, rect.inflate(-30, -40).topright, rect.inflate(-30, -40).bottomleft, 6)
        if clickable is not None:
            self.hits.append((rect, clickable))

    def _draw_hero(
        self,
        center: tuple[int, int],
        player_id: str,
        player: PlayerState | None,
        *,
        radius: int = 30,
        health: int | None = None,
    ) -> None:
        color = (
            HERO_COLORS[self.player_ids.index(player_id) % len(HERO_COLORS)]
            if player_id in self.player_ids
            else (90, 90, 96)
        )
        pygame.draw.circle(self.screen, (20, 14, 10), center, radius + 3)
        pygame.draw.circle(self.screen, color, center, radius)
        pygame.draw.circle(self.screen, GOLD, center, radius, 2)
        self._draw_text(_initials(player_id), radius // 2 + 6, TEXT, center, bold=True, anchor="center", outline=True)
        if player is not None:
            shown = player.health if health is None else health
            gem = (center[0] + radius - 4, center[1] + radius - 6)
            self._draw_gem(gem, shown, (176, 40, 36), TEXT, 13)
            if player.armor:
                self._draw_gem((center[0] - radius + 4, center[1] + radius - 6), player.armor, (120, 120, 130), TEXT, 12)
            self._draw_tier(center[0], center[1] - radius - 8, player.tavern_tier, 5)

    def _draw_leaderboard(self) -> None:
        self._draw_text("Lobby", 18, GOLD, (LEFT_X + 4, 16), bold=True)
        displayed = self._displayed_player_id()
        pending = self.engine.pending_player_ids
        placements = dict(self.engine.placements)
        players = (
            self.combats[0].players_before if self.combats else self.engine.state.players
        )
        fighting = set(_combatants(self.combats[0].result)) if self.combats else set()
        for index, player in enumerate(players):
            rect = pygame.Rect(LEFT_X, TOP_H + 4 + index * 80, LEFT_W, 74)
            is_displayed = player.player_id == displayed
            pygame.draw.rect(self.screen, PANEL, rect, border_radius=10)
            pygame.draw.rect(self.screen, GOLD if is_displayed else TABLE_EDGE, rect, 3 if is_displayed else 1, border_radius=10)
            self._draw_hero((rect.x + 38, rect.centery + 6), player.player_id, player, radius=24)
            label = player.player_id + self._seat_tag(player.player_id)
            self._draw_text(label, 15, TEXT, (rect.x + 74, rect.y + 8), bold=True)
            if player.health <= 0:
                place = placements.get(player.player_id)
                status = f"Eliminated #{place}" if place else "Eliminated"
                color = RED
            elif player.player_id in fighting:
                status, color = "In combat", (255, 150, 120)
            elif self.combats:
                status, color = "Combat", MUTED
            elif self.engine.finished:
                status, color = f"Place #{placements.get(player.player_id, '?')}", GOLD
            elif pending and pending[0] == player.player_id and not self.combats:
                status, color = "Recruiting...", GREEN
            elif player.player_id not in pending and not self.engine.finished:
                status, color = "Ready", MUTED
            else:
                status, color = "Waiting", MUTED
            self._draw_text(status, 13, color, (rect.x + 74, rect.y + 30))
            self._draw_text(f"Tier {player.tavern_tier}", 13, MUTED, (rect.x + 74, rect.y + 48))
            if player.health <= 0:
                shade = pygame.Surface(rect.size, pygame.SRCALPHA)
                shade.fill((0, 0, 0, 110))
                self.screen.blit(shade, rect)

            def view_player(player_id: str = player.player_id) -> None:
                self._set_view(player_id)

            self.hits.append((rect, view_player))

    def _draw_log(self) -> None:
        rect = pygame.Rect(LOG_X, TOP_H, LOG_W, BOTTOM_Y - TOP_H - 6)
        pygame.draw.rect(self.screen, PANEL, rect, border_radius=10)
        pygame.draw.rect(self.screen, TABLE_EDGE, rect, 1, border_radius=10)
        title = "Match log" if self.pov is None else f"Log: {self.pov}"
        self._draw_text(title, 18, GOLD, (LOG_X + 4, 16), bold=True)
        wrapped: list[str] = []
        for line in visible_lines(self.log, self.pov):
            wrapped.extend(self._wrap(line, 13, LOG_W - 18) or [""])
            wrapped.append("")
        line_height = 16
        capacity = (rect.height - 16) // line_height
        self.log_scroll = min(self.log_scroll, max(0, len(wrapped) - capacity))
        end = len(wrapped) - self.log_scroll
        shown = wrapped[max(0, end - capacity) : end]
        y = rect.y + 8
        for line in shown:
            color = GOLD if line.startswith("Combat:") else TEXT if line.startswith("You") else MUTED
            self._draw_text(line, 13, color, (rect.x + 9, y))
            y += line_height

    def _draw_recruit(self) -> None:
        player = self.engine.player_state(self._displayed_player_id())
        can_act = self._can_act()
        is_you = player.player_id == self.human_player_id
        owner = "Your" if is_you else f"{player.player_id}'s"
        legal = set(self.engine.legal_actions(player.player_id)) if can_act else set()

        # Top bar: turn, gold, tier, and tavern buttons.
        phase = "Replay" if self.replaying else "Recruit"
        self._draw_text(f"Turn {self.engine.state.turn}  |  {phase}", 20, TEXT, (MAIN_X, 14), bold=True)
        for index in range(self.rules.max_gold):
            center = (MAIN_X + 196 + index * 13, 26)
            filled = index < player.gold
            pygame.draw.circle(self.screen, GOLD if filled else (70, 60, 44), center, 5)
        self._draw_text(f"{player.gold}/{self.rules.max_gold}", 16, GOLD, (MAIN_X + 192 + self.rules.max_gold * 13, 16), bold=True)
        self._draw_tier(MAIN_X + 420, 26, player.tavern_tier, 6)

        def act(kind: RecruitActionKind) -> Callable[[], None] | None:
            action = RecruitAction(kind)
            return (lambda: self._human_action(action)) if action in legal else None

        x = MAIN_X + 470
        if player.tavern_tier < self.rules.max_tavern_tier:
            cost = self.rules.tavern_upgrade_costs[player.tavern_tier - 1]
            self._button(pygame.Rect(x, 8, 104, 36), f"Upgrade ({cost})", act(RecruitActionKind.UPGRADE_TAVERN))
        x += 110
        self._button(pygame.Rect(x, 8, 92, 36), f"Reroll ({self.rules.reroll_cost})", act(RecruitActionKind.REROLL))
        x += 98
        freeze_kind = RecruitActionKind.UNFREEZE if player.shop_frozen else RecruitActionKind.FREEZE
        self._button(
            pygame.Rect(x, 8, 82, 36),
            "Unfreeze" if player.shop_frozen else "Freeze",
            act(freeze_kind),
            accent=(50, 90, 140),
        )
        # End Turn sits on the right of the table, like the real game.
        self._button(
            pygame.Rect(MAIN_X + MAIN_W - 134, HAND_Y + CARD_H + 14, 126, 46),
            "End Turn",
            act(RecruitActionKind.END_RECRUIT),
            size=18,
            accent=(150, 110, 30),
        )

        # Shop.
        frozen_note = "  (frozen)" if player.shop_frozen else ""
        self._draw_text(f"Bob's Tavern{frozen_note}", 16, FROST if player.shop_frozen else GOLD, (MAIN_X + 6, SHOP_Y - 22), bold=True)
        for rect, offer in zip(_row_rects(len(player.shop), SHOP_Y, CARD_W, CARD_H), player.shop):
            buy = RecruitAction(RecruitActionKind.BUY, offer.offer_id)
            self._draw_card(
                rect,
                self._offer_view(offer),
                frozen=player.shop_frozen,
                clickable=(lambda a=buy: self._human_action(a))
                if buy in legal
                else self._explain(player, "buy") if can_act else None,
            )

        # Board.
        self._draw_text(
            f"{owner} warband  {len(player.board)}/{self.rules.board_limit}  (attacks left to right)",
            16,
            GOLD,
            (MAIN_X + 6, BOARD_Y - 22),
            bold=True,
        )
        board_ids = [minion.instance_id for minion in player.board]
        if self.selected_id not in board_ids:
            self.selected_id = None
        targeted_plays = {
            action.effect_target_id: action
            for action in legal
            if action.kind is RecruitActionKind.PLAY
            and action.target_id == self.targeting_id
            and action.effect_target_id is not None
        }
        if not targeted_plays:
            self.targeting_id = None
        board_rects = _row_rects(len(player.board), BOARD_Y, MINION_W, MINION_H, gap=12)
        for rect, minion in zip(board_rects, player.board):

            def select(instance_id: str = minion.instance_id) -> None:
                self.selected_id = None if self.selected_id == instance_id else instance_id

            play = targeted_plays.get(minion.instance_id)
            if self.targeting_id is not None:
                click = (
                    (lambda a=play: self._human_action(a))
                    if play is not None
                    else self._cancel_targeting
                )
            else:
                click = select if can_act else None
            self._draw_minion(
                rect,
                self._instance_view(minion),
                selected=minion.instance_id == self.selected_id,
                targetable=play is not None,
                clickable=click,
            )
        if not player.board:
            self._draw_text("No minions yet - play one from your hand.", 15, MUTED, (MAIN_X + MAIN_W // 2, BOARD_Y + MINION_H // 2), anchor="center")
        if can_act and self.selected_id is not None and self.targeting_id is None:
            index = board_ids.index(self.selected_id)
            rect = board_rects[index]
            y = rect.bottom + 6
            sell = RecruitAction(RecruitActionKind.SELL, self.selected_id)
            reorder = RecruitAction(RecruitActionKind.REORDER_BOARD)

            def move(offset: int) -> Callable[[], None] | None:
                target = index + offset
                if reorder not in legal or not 0 <= target < len(board_ids):
                    return None

                def apply() -> None:
                    order = list(board_ids)
                    order[index], order[target] = order[target], order[index]
                    self._human_action(reorder, tuple(order))

                return apply

            self._button(pygame.Rect(rect.centerx - 62, y, 30, 26), "<", move(-1))
            self._button(
                pygame.Rect(rect.centerx - 28, y, 56, 26),
                f"Sell +{self.rules.sell_refund}",
                (lambda: self._human_action(sell)) if sell in legal else None,
                size=13,
                accent=(140, 50, 40),
            )
            self._button(pygame.Rect(rect.centerx + 32, y, 30, 26), ">", move(1))

        # Hand.
        self._draw_text(
            f"{owner} hand  {len(player.hand)}/{self.rules.hand_limit}",
            16,
            GOLD,
            (MAIN_X + 6, HAND_Y - 22),
            bold=True,
        )
        for rect, minion in zip(_row_rects(len(player.hand), HAND_Y, CARD_W, CARD_H), player.hand):
            plays = [
                action
                for action in legal
                if action.kind is RecruitActionKind.PLAY
                and action.target_id == minion.instance_id
            ]
            if not plays:
                click = self._explain(player, "play") if can_act else None
            elif plays[0].effect_target_id is None:
                click = lambda a=plays[0]: self._human_action(a)
            else:
                click = lambda instance_id=minion.instance_id: self._start_targeting(instance_id)
            self._draw_card(rect, self._instance_view(minion), clickable=click)
            if minion.instance_id == self.targeting_id:
                pygame.draw.rect(self.screen, GREEN, rect.inflate(6, 6), 4, border_radius=12)

        y = HAND_Y + CARD_H + 18
        for line in self._wrap(self._hint(player, can_act), 15, MAIN_W - 160):
            self._draw_text(line, 15, TEXT, (MAIN_X + 10, y))
            y += 20

    def _start_targeting(self, instance_id: str) -> None:
        self.targeting_id = None if self.targeting_id == instance_id else instance_id
        self.selected_id = None

    def _cancel_targeting(self) -> None:
        self.targeting_id = None

    def _explain(self, player: PlayerState, kind: str) -> Callable[[], None]:
        if kind == "buy":
            reason = (
                "Your hand is full."
                if len(player.hand) >= self.rules.hand_limit
                else f"You need {self.rules.buy_cost} gold to buy."
            )
        else:
            reason = "Your board is full."

        def show() -> None:
            x, y = pygame.mouse.get_pos()
            self._float(reason, (255, 150, 120), x, y - 16, 1400)

        return show

    def _hint(self, player: PlayerState, can_act: bool) -> str:
        if self.engine.finished:
            return "The match is over."
        if can_act and self.targeting_id is not None:
            minion = next(
                (m for m in player.hand if m.instance_id == self.targeting_id), None
            )
            name = self._instance_view(minion).name if minion else "this minion"
            return (
                f"Choose a glowing minion for {name}'s Battlecry. "
                "Right-click, Esc, or click elsewhere to cancel."
            )
        if can_act:
            return (
                "Your turn: click a shop card to buy, a hand card to play, "
                "a board minion to sell or move it. Hover for card text."
            )
        pending = self.engine.pending_player_ids
        if (
            self.human_player_id is not None
            and pending
            and pending[0] == self.human_player_id
            and not self.replaying
        ):
            return "It's your turn - click your hero in the Lobby to act."
        if self.paused:
            return "Paused. Press N (or Step) to advance one action."
        actor = pending[0] if pending else None
        if actor and actor != player.player_id:
            return "Waiting for other players..."
        return f"{player.player_id} is recruiting..."

    def _combat_rects(self, view: CombatView, frame: CombatFrame) -> dict[str, pygame.Rect]:
        rects: dict[str, pygame.Rect] = {}
        for side, board in enumerate(frame.boards):
            y = 382 if side == view.bottom else 150
            for rect, minion in zip(_row_rects(len(board), y, MINION_W, MINION_H, gap=12), board):
                rects[minion.instance_id] = rect
        return rects

    def _hero_center(self, view: CombatView, side: int) -> tuple[int, int]:
        return (MAIN_X + MAIN_W // 2, 600 if side == view.bottom else 96)

    def _draw_combat(self, view: CombatView) -> None:
        result = view.result
        frame = result.frames[view.frame]
        ids = _combatants(result)
        elapsed = self.clock_ms - (view.frame_start or self.clock_ms)
        progress = min(1.0, elapsed / FRAME_DURATIONS.get(frame.kind, 500))
        self._draw_text(f"Turn {view.turn}  |  Combat", 20, TEXT, (MAIN_X, 14), bold=True)
        self._button(pygame.Rect(MAIN_X + MAIN_W - 150, 8, 146, 36), "Skip combat (Esc)", self._skip_combats, size=14)
        remaining = len(self.combats) - 1
        if remaining:
            self._draw_text(f"{remaining} more combat(s) queued", 14, MUTED, (MAIN_X + 300, 18))

        for side in range(2):
            x, y = self._hero_center(view, side)
            hero = view.heroes_after[side] if frame.kind == "end" and progress > 0.3 else view.heroes_before[side]
            self._draw_hero((x, y), ids[side], hero or view.heroes_before[side], radius=34)
            self._draw_text(_hero_label(ids[side]), 15, TEXT, (x + 48, y - 8))

        rects = self._combat_rects(view, frame)
        moving_id = frame.source_instance_id if frame.kind in ("attack", "hit") else None
        target_id = frame.target_instance_id
        views = {
            minion.instance_id: self._combat_minion_view(minion)
            for board in frame.boards
            for minion in board
        }
        for key, rect in rects.items():
            if key != moving_id:
                self._draw_minion(rect, views[key])
        if moving_id in rects and target_id in rects:
            start = pygame.Vector2(rects[moving_id].center)
            end = pygame.Vector2(rects[target_id].center)
            eased = 1 - (1 - progress) ** 2
            reach = eased if frame.kind == "attack" else 1 - eased
            position = start.lerp(end, 0.8 * reach)
            rect = rects[moving_id].copy()
            rect.center = (round(position.x), round(position.y))
            pygame.draw.ellipse(self.screen, GOLD, rect.inflate(12, 12), 3)
            self._draw_minion(rect, views[moving_id])

        center_y = 330
        if frame.kind == "start":
            first = result.first_attacker_id
            self._banner(f"{_hero_label(ids[0])}  vs  {_hero_label(ids[1])}", f"{_hero_label(first or '')} attacks first", center_y)
        elif frame.kind == "end":
            if result.winner_id is None:
                self._banner("Tie!", "No damage dealt.", center_y)
            else:
                self._banner(
                    f"{_hero_label(result.winner_id)} wins!",
                    f"{_hero_label(result.loser_id or '')} takes {result.damage} damage "
                    f"(winner's tier + surviving minion tiers).",
                    center_y,
                )

    def _banner(self, title: str, subtitle: str, center_y: int) -> None:
        rect = pygame.Rect(0, 0, 520, 66)
        rect.center = (MAIN_X + MAIN_W // 2, center_y)
        panel = pygame.Surface(rect.size, pygame.SRCALPHA)
        pygame.draw.rect(panel, (20, 14, 10, 215), panel.get_rect(), border_radius=12)
        pygame.draw.rect(panel, (*GOLD, 255), panel.get_rect(), 2, border_radius=12)
        self.screen.blit(panel, rect)
        self._draw_text(title, 22, GOLD, (rect.centerx, rect.y + 20), bold=True, anchor="center")
        self._draw_text(subtitle, 14, TEXT, (rect.centerx, rect.y + 47), anchor="center")

    def _draw_bottom_bar(self) -> None:
        y = BOTTOM_Y + 4
        self._button(pygame.Rect(MAIN_X, y, 104, 36), "Resume" if self.paused else "Pause", self._toggle_pause)
        self._button(pygame.Rect(MAIN_X + 110, y, 80, 36), "Step (N)", self._request_step if self.paused else None)
        self._button(pygame.Rect(MAIN_X + 204, y, 36, 36), "-", lambda: self._change_speed(-1))
        self._draw_text(f"Speed {SPEEDS[self.speed_index]:g}x", 15, TEXT, (MAIN_X + 290, y + 18), bold=True, anchor="center")
        self._button(pygame.Rect(MAIN_X + 340, y, 36, 36), "+", lambda: self._change_speed(1))
        self._draw_text("View as", 15, MUTED, (MAIN_X + 400, y + 18), anchor="midleft")
        self._button(pygame.Rect(MAIN_X + 462, y, 32, 36), "<", lambda: self._cycle_view(-1))
        self._draw_text(self.view_options[self.view_index], 15, GOLD, (MAIN_X + 556, y + 18), bold=True, anchor="center")
        self._button(pygame.Rect(MAIN_X + 618, y, 32, 36), ">", lambda: self._cycle_view(1))
        if self.engine.finished and self.last_replay is not None and not self.combats:
            self._button(pygame.Rect(LOG_X, y, LOG_W, 36), "Replay match", self._start_replay, accent=(60, 110, 60))
        else:
            self._draw_text("Space pause | N step | +/- speed", 13, MUTED, (LOG_X + LOG_W // 2, y + 18), anchor="center")

    def _draw_floats(self) -> None:
        for text in self.floats:
            age = (self.clock_ms - text.start) / text.duration
            self._draw_text(
                text.text,
                22,
                text.color,
                (text.x, text.y - 40 * age),
                bold=True,
                anchor="center",
                outline=True,
            )

    def _draw_game_over(self) -> None:
        rect = pygame.Rect(0, 0, 360, 80 + 26 * len(self.player_ids))
        rect.center = (MAIN_X + MAIN_W // 2, HEIGHT // 2 - 20)
        panel = pygame.Surface(rect.size, pygame.SRCALPHA)
        pygame.draw.rect(panel, (20, 14, 10, 235), panel.get_rect(), border_radius=14)
        pygame.draw.rect(panel, (*GOLD, 255), panel.get_rect(), 3, border_radius=14)
        self.screen.blit(panel, rect)
        self._draw_text("Match over", 26, GOLD, (rect.centerx, rect.y + 28), bold=True, anchor="center")
        y = rect.y + 62
        for player_id, place in sorted(self.engine.placements, key=lambda item: item[1]):
            label = player_id + self._seat_tag(player_id)
            self._draw_text(f"#{place}  {label}", 17, TEXT if place > 1 else GOLD, (rect.x + 40, y), bold=place == 1)
            y += 26

    def _draw_tooltip(self, view: MinionView) -> None:
        width, height = 210, 290
        mouse_x, mouse_y = pygame.mouse.get_pos()
        x = mouse_x + 18 if mouse_x + 18 + width < WIDTH else mouse_x - width - 18
        y = max(8, min(mouse_y - height // 2, HEIGHT - height - 8))
        rect = pygame.Rect(x, y, width, height)
        pygame.draw.rect(self.screen, (20, 14, 10), rect.inflate(6, 6), border_radius=14)
        pygame.draw.rect(self.screen, CARD_BODY, rect, border_radius=12)
        pygame.draw.rect(self.screen, GOLD, rect, 2, border_radius=12)
        art = pygame.Rect(rect.x + 20, rect.y + 26, rect.width - 40, 86)
        pygame.draw.ellipse(self.screen, TRIBE_COLORS.get(view.tribe or "", NEUTRAL_TRIBE), art)
        pygame.draw.ellipse(self.screen, (214, 176, 98), art, 3)
        self._draw_tier(rect.centerx, rect.y + 14, view.tier, 7)
        self._draw_text(view.name, self._fit_size(view.name, rect.width - 16, 18), TEXT, (rect.centerx, art.bottom + 16), bold=True, anchor="center")
        y_text = art.bottom + 34
        for line in self._wrap(view.text or "No card text.", 14, rect.width - 24):
            self._draw_text(line, 14, TEXT, (rect.centerx, y_text), anchor="midtop")
            y_text += 18
        extras = [keyword.value.replace("_", " ").title() for keyword in sorted(view.keywords)]
        if extras:
            self._draw_text(", ".join(extras), 13, FROST, (rect.centerx, y_text + 4), anchor="midtop")
        if view.tribe:
            self._draw_text(view.tribe, 13, MUTED, (rect.centerx, rect.bottom - 22), anchor="center")
        attack_color, health_color = self._stat_colors(view)
        self._draw_gem((rect.x + 22, rect.bottom - 22), view.attack, (204, 152, 40), attack_color, 18)
        self._draw_gem((rect.right - 22, rect.bottom - 22), view.health, (176, 40, 36), health_color, 18)


def _row_rects(
    count: int,
    y: int,
    width: int,
    height: int,
    *,
    gap: int = 8,
    available: int = MAIN_W,
) -> list[pygame.Rect]:
    """Rects for ``count`` items centred in a row, overlapping if they don't fit."""
    if count > 1 and count * width + (count - 1) * gap > available:
        gap = (available - count * width) // (count - 1)
    total = count * width + max(0, count - 1) * gap
    start = MAIN_X + (available - total) // 2
    return [pygame.Rect(start + index * (width + gap), y, width, height) for index in range(count)]


def _combatants(result: CombatResult) -> tuple[str, str]:
    first, second = (player_id for player_id, _survivors in result.survivors_by_player)
    return first, second


def _hero_label(player_id: str) -> str:
    return "Ghost" if player_id.startswith("ghost:") else player_id


def _initials(player_id: str) -> str:
    if player_id.startswith("ghost:"):
        return "G"
    if player_id.startswith("player-"):
        return "P" + player_id.removeprefix("player-")
    return player_id[:2].upper()


def _tribe(card: CardRecord) -> str | None:
    if card.card_type == "All":
        return "All"
    return card.tribe


def launch_pygame_gui(
    profile: IterationProfile,
    selected_cards: Sequence[CardRecord],
    seed: int,
    *,
    human_player_id: str | None = "player-1",
    pov_player_id: str | None = None,
    agents: Mapping[str, TrainedBot] | None = None,
) -> None:
    """Open the table view for human-vs-bots or spectator play."""
    PygameMatchClient(
        profile, selected_cards, seed, human_player_id, pov_player_id, agents
    ).run()

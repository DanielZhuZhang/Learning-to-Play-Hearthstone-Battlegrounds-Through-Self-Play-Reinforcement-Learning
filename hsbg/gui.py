"""Tkinter client for human play, bot watching, and match replay."""

from __future__ import annotations

import random
import tkinter as tk
from collections.abc import Sequence
from tkinter import ttk

from hsbg.data.models import CardRecord, IterationProfile
from hsbg.game.combat import CombatEvent, CombatResult
from hsbg.game.engine import ActionRecord, GameEngine, MatchReplay
from hsbg.game.legal import RecruitAction, RecruitActionKind
from hsbg.game.state import MinionInstance, PlayerState, ShopOffer
from hsbg.ui_common import (
    ALL_PLAYERS,
    EventLine,
    card_name,
    describe_action,
    minion_name,
    visible_lines,
)


class MatchWindow:
    """Small desktop client that delegates every rule to ``GameEngine``."""

    def __init__(
        self,
        profile: IterationProfile,
        selected_cards: Sequence[CardRecord],
        seed: int,
        human_player_id: str | None,
        pov_player_id: str | None = None,
    ) -> None:
        self.profile = profile
        self.cards = tuple(selected_cards)
        self.seed = seed
        self.human_player_id = human_player_id
        self.engine = GameEngine(profile, self.cards, seed)
        self.bot_rng = random.Random(seed ^ 0xB075)
        self.card_by_id = {card.card_id: card for card in self.cards}
        self.last_replay: MatchReplay | None = None
        self._replay_index = 0
        self._replaying = False
        self.paused = False
        self._pending_event_lines: list[EventLine] = []
        self._bot_action_turn = self.engine.state.turn
        self._bot_action_counts: dict[str, int] = {}
        self._max_bot_actions_per_turn = 32
        self._event_lines: list[EventLine] = [
            (
                "Welcome! During Recruit, spend gold to build your board. "
                "When everyone ends Recruit, minions fight automatically.",
                None,
            )
        ]

        self.root = tk.Tk()
        self.root.title("Battlegrounds Simulator")
        self.root.minsize(980, 720)
        self.status = tk.StringVar()
        self.speed = tk.IntVar(value=400)
        self.pov_choice = tk.StringVar(
            value=pov_player_id or human_player_id or ALL_PLAYERS
        )
        self.status_label = ttk.Label(
            self.root, textvariable=self.status, font=("Segoe UI", 12, "bold")
        )
        self.status_label.pack(fill=tk.X, padx=12, pady=(12, 6))
        self.help_text = ttk.Label(
            self.root,
            text=(
                "HOW TO PLAY  |  Buy minions from the shop, then Play them from "
                "your hand onto your board. You may also sell minions, reroll or "
                "freeze the shop, or upgrade your Tavern. Press End Recruit when "
                "ready. Once everyone is ready, combat is automatic: the larger "
                "board starts (ties are random), then attacks alternate. Each "
                "board attacks from left to right. Armor absorbs damage before "
                "Health. Tavern upgrades unlock higher-tier minions. Card text "
                "appears beside each minion."
            ),
            wraplength=940,
            justify=tk.LEFT,
        )
        self.help_text.pack(fill=tk.X, padx=12, pady=6)
        self.lobby_frame = ttk.LabelFrame(self.root, text="Lobby")
        self.lobby_frame.pack(fill=tk.X, padx=12, pady=6)
        self.lobby = tk.Text(
            self.lobby_frame, height=7, wrap=tk.WORD, state=tk.DISABLED
        )
        self.lobby.pack(fill=tk.X, padx=6, pady=6)
        self.player_frame = ttk.LabelFrame(self.root, text="Your tavern")
        self.player_frame.pack(fill=tk.X, padx=12, pady=6)
        self.player_view = tk.Text(
            self.player_frame, height=9, wrap=tk.WORD, state=tk.DISABLED
        )
        self.player_view.pack(fill=tk.X, padx=6, pady=6)
        self.action_frame = ttk.LabelFrame(self.root, text="Recruit actions")
        self.action_frame.pack(fill=tk.X, padx=12, pady=6)
        self.event_frame = ttk.LabelFrame(self.root, text="What just happened")
        self.event_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=6)
        self.event_log = tk.Text(
            self.event_frame, height=8, wrap=tk.WORD, state=tk.DISABLED
        )
        self.event_log.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        controls = ttk.Frame(self.root)
        controls.pack(fill=tk.X, padx=12, pady=(6, 12))
        self.pause_button = ttk.Button(
            controls, text="Pause", command=self._toggle_pause
        )
        self.pause_button.pack(side=tk.LEFT)
        self.step_button = ttk.Button(
            controls, text="Single step", command=self._single_step
        )
        self.step_button.pack(side=tk.LEFT, padx=6)
        ttk.Label(controls, text="Playback speed (ms)").pack(side=tk.LEFT, padx=(12, 4))
        ttk.Scale(
            controls,
            from_=50,
            to=1500,
            variable=self.speed,
            orient=tk.HORIZONTAL,
            length=180,
        ).pack(side=tk.LEFT)
        ttk.Label(controls, text="View as").pack(side=tk.LEFT, padx=(12, 4))
        pov_selector = ttk.Combobox(
            controls,
            textvariable=self.pov_choice,
            values=(ALL_PLAYERS,)
            + tuple(player.player_id for player in self.engine.state.players),
            state="readonly",
            width=12,
        )
        pov_selector.pack(side=tk.LEFT)
        pov_selector.bind("<<ComboboxSelected>>", lambda _event: self._render())
        self.replay_button = ttk.Button(
            controls, text="Replay last match", command=self._start_replay
        )
        self.replay_button.pack(side=tk.RIGHT)
        self._render()
        self.root.after(0, self._tick)

    def run(self) -> None:
        """Enter Tk's event loop."""
        self.root.mainloop()

    @property
    def _pov(self) -> str | None:
        choice = self.pov_choice.get()
        return None if choice == ALL_PLAYERS else choice

    def _flush_hidden_pending(self) -> None:
        pov = self._pov
        while self._pending_event_lines and not visible_lines(
            self._pending_event_lines[:1], pov
        ):
            self._event_lines.append(self._pending_event_lines.pop(0))

    def _tick(self) -> None:
        self._flush_hidden_pending()
        if self._pending_event_lines and not self.paused:
            self._event_lines.append(self._pending_event_lines.pop(0))
            self._flush_hidden_pending()
            self._render()
            self.root.after(max(self.speed.get(), 1), self._tick)
            return
        if self.engine.finished:
            self._render()
            return
        if not self.paused:
            advanced = self._step()
            self._render()
            pending = self.engine.pending_player_ids
            bot_is_next = bool(pending and pending[0] != self.human_player_id)
            if advanced and (
                self._pending_event_lines or self._replaying or bot_is_next
            ):
                self.root.after(max(self.speed.get(), 1), self._tick)
        elif self._replaying:
            return

    def _step(self) -> bool:
        if self._replaying:
            if self.last_replay is None:
                raise RuntimeError("Replay mode has no saved match.")
            if self._replay_index >= len(self.last_replay.actions):
                self._replaying = False
                return False
            record = self.last_replay.actions[self._replay_index]
            self._replay_index += 1
            player_before = self.engine.player_state(record.player_id)
            before_combats = len(self.engine.combat_history)
            self.engine.apply_action(
                record.player_id,
                record.action,
                board_order=record.board_order,
            )
            self._log_action(record, player_before)
            self._log_new_combats(before_combats)
            return True

        pending = self.engine.pending_player_ids
        if not pending:
            if self.engine.finished:
                return False
            raise RuntimeError("Match has no pending player but is not finished.")
        player_id = pending[0]
        if player_id == self.human_player_id:
            return False
        if self.engine.state.turn != self._bot_action_turn:
            self._bot_action_turn = self.engine.state.turn
            self._bot_action_counts.clear()
        action = self._choose_bot_action(player_id)
        board_order = self._random_board_order(player_id, action)
        record = ActionRecord(player_id, action, board_order)
        player_before = self.engine.player_state(player_id)
        before_combats = len(self.engine.combat_history)
        self.engine.apply_action(player_id, action, board_order=board_order)
        if action.kind is not RecruitActionKind.END_RECRUIT:
            self._bot_action_counts[player_id] = (
                self._bot_action_counts.get(player_id, 0) + 1
            )
        self._log_action(record, player_before)
        self._log_new_combats(before_combats)
        if self.engine.finished:
            self.last_replay = self.engine.to_replay()
        return True

    def _choose_bot_action(self, player_id: str) -> RecruitAction:
        legal_actions = self.engine.legal_actions(player_id)
        if self._bot_action_counts.get(player_id, 0) >= self._max_bot_actions_per_turn:
            return next(
                action
                for action in legal_actions
                if action.kind is RecruitActionKind.END_RECRUIT
            )
        action = self.bot_rng.choice(legal_actions)
        return action

    def _random_board_order(
        self,
        player_id: str,
        action: RecruitAction,
    ) -> tuple[str, ...] | None:
        if action.kind is not RecruitActionKind.REORDER_BOARD:
            return None
        board = self.engine.player_state(player_id).board
        return tuple(
            minion.instance_id
            for minion in self.bot_rng.sample(list(board), len(board))
        )

    def _perform_human_action(self, action: RecruitAction) -> None:
        if self.human_player_id is None:
            raise RuntimeError("Spectator mode does not accept human actions.")
        if not self.engine.pending_player_ids or (
            self.engine.pending_player_ids[0] != self.human_player_id
        ):
            raise RuntimeError("The human player is not currently recruiting.")
        player_before = self.engine.player_state(self.human_player_id)
        board_order = None
        if action.kind is RecruitActionKind.REORDER_BOARD:
            board_order = tuple(
                minion.instance_id
                for minion in reversed(
                    self.engine.player_state(self.human_player_id).board
                )
            )
        record = ActionRecord(self.human_player_id, action, board_order)
        before_combats = len(self.engine.combat_history)
        self.engine.apply_action(self.human_player_id, action, board_order=board_order)
        self._log_action(record, player_before)
        self._log_new_combats(before_combats)
        if self.engine.finished:
            self.last_replay = self.engine.to_replay()
        self._render()
        if not self.engine.finished or self._pending_event_lines:
            self.root.after(0, self._tick)

    def _toggle_pause(self) -> None:
        self.paused = not self.paused
        self.pause_button.configure(text="Resume" if self.paused else "Pause")
        if not self.paused:
            self.root.after(0, self._tick)

    def _single_step(self) -> None:
        if not self.paused:
            return
        self._flush_hidden_pending()
        if self._pending_event_lines:
            self._event_lines.append(self._pending_event_lines.pop(0))
            self._flush_hidden_pending()
            self._render()
            return
        self._step()
        self._render()

    def _start_replay(self) -> None:
        if self.last_replay is None:
            return
        replay = self.last_replay
        self.engine = GameEngine(
            self.profile,
            self.cards,
            replay.seed,
            replay.player_ids,
        )
        self._replay_index = 0
        self._replaying = True
        self.paused = False
        self._pending_event_lines.clear()
        self._event_lines = [
            ("Replaying the recorded match from its first action.", None)
        ]
        self.pause_button.configure(text="Pause")
        self._render()
        self.root.after(0, self._tick)

    def _render(self) -> None:
        pov = self._pov
        if self.engine.finished:
            placement_text = ", ".join(
                f"{player_id}: {place}" for player_id, place in self.engine.placements
            )
            self.status.set(f"Match finished | Placements: {placement_text}")
        else:
            actor = (
                self.engine.pending_player_ids[0]
                if self.engine.pending_player_ids
                else "none"
            )
            if pov is not None and actor not in {pov, "none"}:
                actor = "waiting for other players"
            phase = "replay" if self._replaying else self.engine.state.phase.value
            self.status.set(f"Turn {self.engine.state.turn} | {phase} | Next: {actor}")

        lines = []
        for player in self.engine.state.players:
            status = "eliminated" if player.health <= 0 else "alive"
            if pov is not None and player.player_id != pov:
                lines.append(
                    f"{player.player_id:10} {status:10}  Health {player.health:>3}  "
                    f"Armor {player.armor:>2}  Tavern {player.tavern_tier}"
                )
                continue
            marker = ""
            if pov is not None:
                marker = "  <- you" if pov == self.human_player_id else "  <- viewing"
            lines.append(
                f"{player.player_id:10} {status:10}  Health {player.health:>3}  "
                f"Armor {player.armor:>2}  Gold {player.gold:>2}  "
                f"Tavern {player.tavern_tier}  Board {len(player.board)}/"
                f"{self.profile.rules.board_limit}{marker}"
            )
        self.lobby.configure(state=tk.NORMAL)
        self.lobby.delete("1.0", tk.END)
        self.lobby.insert(tk.END, "\n".join(lines))
        self.lobby.configure(state=tk.DISABLED)
        self._render_player()
        self._render_events()
        self._render_actions()

    def _render_player(self) -> None:
        viewed_id = self._pov or self.human_player_id
        self.player_frame.configure(
            text="Your tavern"
            if viewed_id is None or viewed_id == self.human_player_id
            else f"{viewed_id}'s tavern"
        )
        if viewed_id is None:
            text = "Spectator mode: watch the lobby above; combats are explained below."
        else:
            player = self.engine.player_state(viewed_id)
            rules = self.profile.rules
            next_upgrade = (
                rules.tavern_upgrade_costs[player.tavern_tier - 1]
                if player.tavern_tier < rules.max_tavern_tier
                else None
            )
            upgrade = (
                f"{next_upgrade} gold" if next_upgrade is not None else "maximum tier"
            )
            text = (
                f"Health: {player.health}    Armor: {player.armor}    "
                f"Gold: {player.gold}/{rules.max_gold}    "
                f"Tavern Tier: {player.tavern_tier} "
                f"(next upgrade: {upgrade})\n"
                f"SHOP — buy cost: {rules.buy_cost} gold\n"
                f"{self._format_shop(player.shop)}\n"
                f"HAND — minions you bought; Play moves them onto your board\n"
                f"{self._format_minions(player.hand)}\n"
                f"YOUR BOARD — attacks follow this left-to-right order; "
                f"{len(player.board)}/{rules.board_limit} spaces used\n"
                f"{self._format_minions(player.board)}"
            )
        self.player_view.configure(state=tk.NORMAL)
        self.player_view.delete("1.0", tk.END)
        self.player_view.insert(tk.END, text)
        self.player_view.configure(state=tk.DISABLED)

    def _render_events(self) -> None:
        self.event_log.configure(state=tk.NORMAL)
        self.event_log.delete("1.0", tk.END)
        self.event_log.insert(
            tk.END, "\n".join(visible_lines(self._event_lines, self._pov)[-100:])
        )
        self.event_log.see(tk.END)
        self.event_log.configure(state=tk.DISABLED)

    def _log_action(
        self,
        record: ActionRecord,
        player_before: PlayerState | None = None,
    ) -> None:
        player = player_before or self.engine.player_state(record.player_id)
        message = describe_action(
            record, player, self.profile.rules, self.card_by_id, self.human_player_id
        )
        self._append_event(message, frozenset({record.player_id}))

    def _log_new_combats(self, previous_count: int) -> None:
        for result in self.engine.combat_history[previous_count:]:
            self._log_combat(result)

    def _log_combat(self, result: CombatResult) -> None:
        lines: list[str] = []
        attacker = result.first_attacker_id or "Neither side"
        if result.winner_id is None:
            lines.append(
                f"COMBAT: {result.survivors_by_player[0][0]} and "
                f"{result.survivors_by_player[1][0]} tied; no hero damage."
            )
        else:
            lines.append(
                f"COMBAT: {attacker} attacked first. {result.winner_id} won; "
                f"{result.loser_id} took {result.damage} damage."
            )
        for event in result.events:
            description = self._describe_combat_event(event)
            if description is not None:
                lines.append(f"  {description}")
        for player_id, survivors in result.survivors_by_player:
            names = (
                ", ".join(
                    f"{minion.token_name or 'summoned token' if minion.token else self._card_name(minion.definition_id)} "
                    f"({minion.attack}/{minion.health})"
                    for minion in survivors
                )
                or "no surviving minions"
            )
            lines.append(f"  {player_id} survivors: {names}.")
        combatants = frozenset(
            player_id for player_id, _survivors in result.survivors_by_player
        )
        self._pending_event_lines.extend((line, combatants) for line in lines)

    def _describe_combat_event(self, event: CombatEvent) -> str | None:
        source = self._instance_name(event.source_instance_id)
        target = (
            self._instance_name(event.target_instance_id)
            if event.target_instance_id is not None
            else None
        )
        if event.kind == "attack" and target is not None:
            return f"{source} attacked {target}."
        if event.kind == "shield" and target is not None:
            return f"{target}'s Divine Shield absorbed the attack."
        if event.kind == "poison" and target is not None:
            return f"{target} was destroyed by Poisonous."
        if event.kind == "death":
            return f"{source} was destroyed."
        if event.kind == "summon" and target is not None:
            return f"{source} summoned {target}."
        if event.kind == "buff" and target is not None:
            return f"{target} received a combat buff from {source}."
        if event.kind == "hero_damage" and target is not None:
            return f"{source} dealt {event.amount} damage to {target}."
        if event.kind == "stalemate":
            return "The combat stalled; no minions could make progress."
        return None

    def _append_event(
        self, message: str, players: frozenset[str] | None = None
    ) -> None:
        self._event_lines.append((message, players))

    def _render_actions(self) -> None:
        for child in self.action_frame.winfo_children():
            child.destroy()
        self.replay_button.configure(
            state=tk.NORMAL
            if self.engine.finished and self.last_replay
            else tk.DISABLED
        )
        if self.engine.finished or self.human_player_id is None:
            return
        if self._pending_event_lines:
            ttk.Label(
                self.action_frame,
                text="Combat recap in progress. Pause or step through the events below.",
            ).grid(row=0, column=0, sticky=tk.W, padx=6, pady=4)
            return
        pending = self.engine.pending_player_ids
        if not pending or pending[0] != self.human_player_id:
            ttk.Label(self.action_frame, text="Waiting for bot actions...").pack(
                anchor=tk.W, padx=6, pady=4
            )
            return
        for index, action in enumerate(self.engine.legal_actions(self.human_player_id)):

            def perform(selected: RecruitAction = action) -> None:
                self._perform_human_action(selected)

            ttk.Button(
                self.action_frame,
                text=self._action_label(action),
                command=perform,
            ).grid(row=index // 3, column=index % 3, sticky=tk.EW, padx=3, pady=4)
        for column in range(3):
            self.action_frame.columnconfigure(column, weight=1)

    def _action_label(self, action: RecruitAction) -> str:
        player = self.engine.player_state(self.human_player_id or "")
        rules = self.profile.rules
        if action.kind is RecruitActionKind.BUY:
            offer = next(
                offer for offer in player.shop if offer.offer_id == action.target_id
            )
            card = self.card_by_id[offer.definition_id]
            return (
                f"Buy {card.name} ({card.attack}/{card.health}) - {rules.buy_cost} gold"
            )
        if action.kind is RecruitActionKind.PLAY:
            minion = next(
                minion
                for minion in player.hand
                if minion.instance_id == action.target_id
            )
            target = next(
                (
                    entry
                    for entry in player.board
                    if entry.instance_id == action.effect_target_id
                ),
                None,
            )
            if target is not None:
                return f"Play {self._minion_name(minion)} on {self._minion_name(target)}"
            return f"Play {self._minion_name(minion)}"
        if action.kind is RecruitActionKind.SELL:
            minion = next(
                minion
                for minion in player.board
                if minion.instance_id == action.target_id
            )
            return f"Sell {self._minion_name(minion)} (+{rules.sell_refund} gold)"
        if action.kind is RecruitActionKind.REORDER_BOARD:
            return "Reverse board order"
        if action.kind is RecruitActionKind.REROLL:
            return f"Reroll shop ({rules.reroll_cost} gold)"
        if action.kind is RecruitActionKind.FREEZE:
            return "Freeze shop (keep these offers next turn)"
        if action.kind is RecruitActionKind.UNFREEZE:
            return "Unfreeze shop"
        if action.kind is RecruitActionKind.UPGRADE_TAVERN:
            return f"Upgrade Tavern ({rules.tavern_upgrade_costs[player.tavern_tier - 1]} gold)"
        if action.kind is RecruitActionKind.END_RECRUIT:
            return "End Recruit (ready for combat)"
        return action.kind.value.replace("_", " ").title()

    def _card_name(self, card_id: str) -> str:
        return card_name(card_id, self.card_by_id)

    def _format_shop(self, offers: Sequence[ShopOffer]) -> str:
        if not offers:
            return "No minions are available."
        return "    ".join(
            f"{index}. {self._card_description(offer.definition_id, self._card_stats(offer.definition_id, offer.attack_bonus, offer.health_bonus))}"
            for index, offer in enumerate(offers, start=1)
        )

    def _format_minions(self, minions: Sequence[MinionInstance]) -> str:
        if not minions:
            return "    (empty)"
        return "  >  ".join(
            f"{index}. {self._minion_description(minion)}"
            for index, minion in enumerate(minions, start=1)
        )

    def _card_stats(self, card_id: str, attack_bonus: int, health_bonus: int) -> str:
        card = self.card_by_id[card_id]
        return f"{card.attack + attack_bonus}/{card.health + health_bonus}"

    def _minion_stats(self, minion: MinionInstance) -> str:
        if minion.token_spec is not None:
            attack = minion.token_spec.attack + minion.attack_bonus
            health = minion.token_spec.health + minion.health_bonus
        else:
            card = self.card_by_id[minion.definition_id]
            attack = card.attack + minion.attack_bonus
            health = card.health + minion.health_bonus
        return f"{attack}/{health}"

    def _minion_name(self, minion: MinionInstance) -> str:
        return minion_name(minion, self.card_by_id)

    def _minion_description(self, minion: MinionInstance) -> str:
        if minion.token_spec is not None:
            return f"{self._minion_name(minion)} ({self._minion_stats(minion)})"
        return self._card_description(
            minion.definition_id,
            self._minion_stats(minion),
        )

    def _card_description(self, card_id: str, stats: str) -> str:
        card = self.card_by_id[card_id]
        description = f"{card.name} ({stats})"
        if card.text.strip():
            description += f" - {card.text.strip()}"
        return description

    def _instance_name(self, instance_id: str) -> str:
        for player in self.engine.state.players:
            if player.player_id == instance_id:
                return player.player_id
            for minion in player.minions:
                if minion.instance_id == instance_id:
                    return self._minion_name(minion)
        return "a summoned minion"


def launch_gui(
    profile: IterationProfile,
    selected_cards: Sequence[CardRecord],
    seed: int,
    *,
    human_player_id: str | None = "player-1",
    pov_player_id: str | None = None,
) -> None:
    """Launch human-versus-random-bots or all-random spectator mode."""
    MatchWindow(
        profile, selected_cards, seed, human_player_id, pov_player_id
    ).run()

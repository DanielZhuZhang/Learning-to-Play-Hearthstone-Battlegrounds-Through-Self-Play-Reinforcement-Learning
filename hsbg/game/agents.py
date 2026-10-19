"""Baseline agents and bounded headless match execution."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Protocol

from hsbg.data.models import CardRecord, IterationProfile
from hsbg.game.engine import GameEngine
from hsbg.game.legal import RecruitAction, RecruitActionKind


class ActionAgent(Protocol):
    """Policy interface shared by bots and tests."""

    def choose_action(
        self,
        legal_actions: Sequence[RecruitAction],
        rng: random.Random,
    ) -> RecruitAction:
        """Choose one action from the supplied legal-action snapshot."""


class RandomLegalAgent:
    """Choose uniformly among the currently exposed recruit operations."""

    def choose_action(
        self,
        legal_actions: Sequence[RecruitAction],
        rng: random.Random,
    ) -> RecruitAction:
        if not legal_actions:
            raise ValueError("RandomLegalAgent requires at least one legal action.")
        return rng.choice(tuple(legal_actions))


def run_headless_match(
    engine: GameEngine,
    agents: Mapping[str, ActionAgent] | None = None,
    *,
    max_actions_per_player_turn: int = 64,
) -> GameEngine:
    """Run a match to completion, ending recruitment when an agent exceeds its cap."""
    if (
        isinstance(max_actions_per_player_turn, bool)
        or not isinstance(max_actions_per_player_turn, int)
        or max_actions_per_player_turn < 1
    ):
        raise ValueError("max_actions_per_player_turn must be a positive integer.")
    default_agent = RandomLegalAgent()
    policies = agents or {}
    policy_rng = random.Random(engine.seed ^ 0x5EED5EED)

    while not engine.finished:
        turn = engine.state.turn
        for player in engine.state.players:
            player_id = player.player_id
            if player.health <= 0 or player_id not in engine.pending_player_ids:
                continue
            agent = policies.get(player_id, default_agent)
            decision_count = 0
            while (
                not engine.finished
                and engine.state.turn == turn
                and player_id in engine.pending_player_ids
            ):
                legal_actions = engine.legal_actions(player_id)
                if decision_count >= max_actions_per_player_turn:
                    action = next(
                        item
                        for item in legal_actions
                        if item.kind is RecruitActionKind.END_RECRUIT
                    )
                else:
                    action = agent.choose_action(legal_actions, policy_rng)
                board_order: tuple[str, ...] | None = None
                if action.kind is RecruitActionKind.REORDER_BOARD:
                    board_order = tuple(
                        minion.instance_id
                        for minion in policy_rng.sample(
                            list(engine.player_state(player_id).board),
                            len(engine.player_state(player_id).board),
                        )
                    )
                engine.apply_action(player_id, action, board_order=board_order)
                decision_count += 1
    return engine


def run_random_match(
    profile: IterationProfile,
    selected_cards: Sequence[CardRecord],
    seed: int,
    player_ids: Sequence[str] | None = None,
    *,
    max_actions_per_player_turn: int = 64,
) -> GameEngine:
    """Create and run a reproducible all-random baseline match."""
    engine = GameEngine(profile, selected_cards, seed, player_ids)
    return run_headless_match(
        engine,
        max_actions_per_player_turn=max_actions_per_player_turn,
    )

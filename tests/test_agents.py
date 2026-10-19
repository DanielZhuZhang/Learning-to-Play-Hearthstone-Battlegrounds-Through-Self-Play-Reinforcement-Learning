from dataclasses import replace
from pathlib import Path
import random

import pytest

from hsbg.data.profiles import load_profile
from hsbg.data.selection import select_enabled_cards
from hsbg.data.workbook import load_sheet
from hsbg.game.agents import RandomLegalAgent, run_headless_match, run_random_match
from hsbg.game.engine import GameEngine
from hsbg.game.legal import RecruitAction, RecruitActionKind

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _content():
    profile = load_profile(PROJECT_ROOT / "data" / "profiles" / "initial-15.6.json")
    cards = select_enabled_cards(
        profile,
        load_sheet(PROJECT_ROOT / profile.workbook, profile.sheet),
    )
    return profile, cards


def test_random_agent_chooses_only_from_available_legal_actions() -> None:
    agent = RandomLegalAgent()
    options = (RecruitAction(RecruitActionKind.END_RECRUIT),)

    assert agent.choose_action(options, random.Random(1)) == options[0]
    with pytest.raises(ValueError, match="at least one legal action"):
        agent.choose_action((), random.Random(1))


def test_headless_runner_ends_turn_when_action_limit_is_reached() -> None:
    profile, cards = _content()
    short_profile = replace(
        profile,
        rules=replace(profile.rules, max_match_turns=1),
    )
    engine = GameEngine(short_profile, cards, seed=14)

    class AlwaysContinueAgent:
        def choose_action(self, legal_actions, rng):
            del rng
            return next(
                action
                for action in legal_actions
                if action.kind is not RecruitActionKind.END_RECRUIT
            )

    run_headless_match(
        engine,
        {player.player_id: AlwaysContinueAgent() for player in engine.state.players},
        max_actions_per_player_turn=1,
    )

    assert engine.finished
    assert len(engine.action_history) == short_profile.rules.player_count * 2
    assert all(
        record.action.kind is RecruitActionKind.END_RECRUIT
        for record in engine.action_history[1::2]
    )


def test_random_match_replay_reproduces_state_and_placements() -> None:
    profile, cards = _content()
    short_profile = replace(
        profile,
        rules=replace(
            profile.rules,
            player_count=4,
            starting_health=1,
            max_match_turns=2,
        ),
    )

    engine = run_random_match(
        short_profile,
        cards,
        seed=29,
        max_actions_per_player_turn=3,
    )
    replay = GameEngine.replay(short_profile, cards, engine.to_replay())

    assert engine.finished
    assert len(engine.placements) == short_profile.rules.player_count
    assert replay.state == engine.state
    assert replay.pool == engine.pool
    assert replay.combat_history == engine.combat_history
    assert replay.placements == engine.placements


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_headless_runner_rejects_invalid_action_limits(limit) -> None:
    profile, cards = _content()
    engine = GameEngine(profile, cards, seed=15)

    with pytest.raises(ValueError, match="positive integer"):
        run_headless_match(engine, max_actions_per_player_turn=limit)

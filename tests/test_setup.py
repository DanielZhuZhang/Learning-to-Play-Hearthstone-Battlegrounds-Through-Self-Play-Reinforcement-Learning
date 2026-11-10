import random
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from hsbg.data.models import CardRecord
from hsbg.data.profiles import load_game_data
from hsbg.game.config import GameRules
from hsbg.game.pool import MinionPool, MinionPoolEntry
from hsbg.game.setup import MatchSetupError, create_initial_match
from hsbg.game.state import MatchPhase, MatchState

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class SetupInputs:
    selected_cards: tuple[CardRecord, ...]
    rules: GameRules
    player_ids: tuple[str, ...]
    health: int | None
    armor: int | None
    offer_ids: tuple[tuple[str, ...], ...]


def _card(card_id: str, tier: int) -> CardRecord:
    number = int(card_id.split(":")[1])
    return CardRecord(
        card_id=card_id,
        tier=tier,
        name=f"Card {number}",
        tribe="",
        card_type="",
        attack=1,
        health=1,
        text="",
    )


def _inputs(
    *,
    rules: GameRules | None = None,
    selected_cards: tuple[CardRecord, ...] | None = None,
    player_ids: tuple[str, ...] = ("p1", "p2", "p3"),
    offer_ids: tuple[tuple[str, ...], ...] = (
        ("p1-1", "p1-2", "p1-3"),
        ("p2-1", "p2-2", "p2-3"),
        ("p3-1", "p3-2", "p3-3"),
    ),
    health: int | None = None,
    armor: int | None = None,
) -> SetupInputs:
    return SetupInputs(
        selected_cards=(
            selected_cards
            if selected_cards is not None
            else (_card("test:1", 1), _card("test:2", 2))
        ),
        rules=rules if rules is not None else replace(GameRules(), player_count=3),
        player_ids=player_ids,
        health=health,
        armor=armor,
        offer_ids=offer_ids,
    )


def _create(inputs: SetupInputs, rng: random.Random) -> tuple[MatchState, MinionPool]:
    return create_initial_match(
        inputs.selected_cards,
        inputs.rules,
        inputs.player_ids,
        inputs.offer_ids,
        rng,
        starting_health=inputs.health,
        starting_armor=inputs.armor,
    )


def test_initial_match_deals_tier_eligible_shops_from_shared_pool() -> None:
    inputs = _inputs(
        selected_cards=(_card("test:1", 1), _card("test:2", 2)),
        offer_ids=(
            ("p1-1", "p1-2", "p1-3"),
            ("p2-1", "p2-2", "p2-3"),
            ("p3-1", "p3-2", "p3-3"),
        ),
    )

    match, pool = _create(inputs, random.Random(13))

    assert match.turn == 1
    assert match.phase is MatchPhase.RECRUIT
    assert tuple(player.player_id for player in match.players) == ("p1", "p2", "p3")
    assert all(player.health == 30 and player.armor == 0 for player in match.players)
    assert all(player.gold == 3 and player.tavern_tier == 1 for player in match.players)
    assert all(len(player.shop) == 3 for player in match.players)
    assert all(
        offer.definition_id == "test:1"
        for player in match.players
        for offer in player.shop
    )
    assert pool.entries == (
        MinionPoolEntry("test:1", tier=1, total_copies=16, available_copies=7),
        MinionPoolEntry("test:2", tier=2, total_copies=15, available_copies=15),
    )


def test_initial_match_setup_is_reproducible_for_a_seed() -> None:
    inputs = _inputs(
        selected_cards=(
            _card("test:1", 1),
            _card("test:2", 1),
            _card("test:3", 1),
        ),
        offer_ids=(
            ("p1-1", "p1-2", "p1-3"),
            ("p2-1", "p2-2", "p2-3"),
            ("p3-1", "p3-2", "p3-3"),
        ),
    )

    first_match, first_pool = _create(inputs, random.Random(42))
    second_match, second_pool = _create(inputs, random.Random(42))

    assert first_match == second_match
    assert first_pool == second_pool


def test_initial_match_uses_the_selected_profile_cards() -> None:
    profile, selected_cards = load_game_data(PROJECT_ROOT / "data" / "profiles" / "initial-15.6.json")
    player_ids = tuple(f"player-{index}" for index in range(8))
    offer_ids = tuple(
        tuple(f"{player_id}-offer-{slot}" for slot in range(1, 4))
        for player_id in player_ids
    )

    match, pool = create_initial_match(
        selected_cards,
        profile.rules,
        player_ids,
        offer_ids_by_player=offer_ids,
        rng=random.Random(2026),
    )

    tier_one_ids = {card.card_id for card in selected_cards if card.tier == 1}
    assert len(match.players) == profile.rules.player_count == 8
    assert all(len(player.shop) == 3 for player in match.players)
    assert all(
        offer.definition_id in tier_one_ids
        for player in match.players
        for offer in player.shop
    )
    assert pool.total_copies == 382
    assert pool.available_copies == 358


@pytest.mark.parametrize(
    ("inputs", "message"),
    [
        (
            replace(_inputs(), player_ids=("p1", "p1", "p3")),
            "Player IDs must be unique",
        ),
        (replace(_inputs(), player_ids=("p1", "p2")), "Expected 3 player IDs"),
        (
            replace(_inputs(), offer_ids=(("p1-1",), ("p2-1",))),
            "Expected offer IDs for 3 players",
        ),
        (
            replace(_inputs(), offer_ids=(("p1-1", "p1-1"), ("p2-1",), ("p3-1",))),
            "must be unique",
        ),
        (replace(_inputs(), health=0), "Starting health must be a positive integer"),
        (replace(_inputs(), armor=-1), "Starting armor must be a nonnegative integer"),
    ],
)
def test_initial_match_rejects_invalid_lobby_inputs(
    inputs: SetupInputs, message: str
) -> None:
    with pytest.raises(MatchSetupError, match=message):
        _create(inputs, random.Random(1))

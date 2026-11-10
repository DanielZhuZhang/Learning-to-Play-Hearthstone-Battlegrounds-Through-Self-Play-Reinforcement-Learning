from pathlib import Path

import pytest

pygame = pytest.importorskip("pygame")

from hsbg.data.profiles import load_game_data
from hsbg.game.legal import RecruitActionKind

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def client_factory(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    from hsbg.pygame_gui import PygameMatchClient

    profile, cards = load_game_data(PROJECT_ROOT / "data" / "profiles" / "initial-15.6.json")
    clients = []

    def make(**kwargs):
        client = PygameMatchClient(profile, cards, **kwargs)
        clients.append(client)
        return client

    yield make
    pygame.quit()


def test_spectator_pov_match_plays_and_animates_to_the_end(client_factory) -> None:
    client = client_factory(seed=7, human_player_id=None, pov_player_id="player-3")
    seen_frames: set[str] = set()
    for _ in range(20000):
        if client.engine.finished and not client.combats:
            break
        client.update(400)
        if client.combats and client.combats[0].frame_start is not None:
            view = client.combats[0]
            seen_frames.add(view.result.frames[view.frame].kind)
            assert "player-3" in {pid for pid, _ in view.result.survivors_by_player}
        client.draw()
    assert client.engine.finished
    assert {"start", "attack", "hit", "end"} <= seen_frames
    assert client.last_replay is not None


def test_clicking_shop_then_hand_buys_and_plays_a_minion(client_factory) -> None:
    from hsbg.pygame_gui import CARD_H, CARD_W, HAND_Y, SHOP_Y, _row_rects

    client = client_factory(seed=3, human_player_id="player-1")
    client.update(16)
    client.draw()

    def click(pos) -> None:
        event = pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=pos)
        client.handle_events([event])
        client.draw()

    player = client.engine.player_state("player-1")
    click(_row_rects(len(player.shop), SHOP_Y, CARD_W, CARD_H)[0].center)
    player = client.engine.player_state("player-1")
    assert len(player.hand) == 1
    click(_row_rects(len(player.hand), HAND_Y, CARD_W, CARD_H)[0].center)
    player = client.engine.player_state("player-1")
    assert len(player.board) == 1 and not player.hand
    kinds = [record.action.kind for record in client.engine.action_history]
    assert kinds == [RecruitActionKind.BUY, RecruitActionKind.PLAY]


def test_targeted_battlecry_waits_for_a_board_click(client_factory) -> None:
    from dataclasses import replace

    from hsbg.game.state import MinionInstance, MinionZone
    from hsbg.pygame_gui import BOARD_Y, CARD_H, CARD_W, HAND_Y, MINION_H, MINION_W, _row_rects

    client = client_factory(seed=3, human_player_id="player-1")
    engine = client.engine
    player = engine.player_state("player-1")
    custom = replace(
        player,
        minions=(
            MinionInstance("demon-1", "15.6:3", MinionZone.BOARD),
            MinionInstance("demon-2", "15.6:3", MinionZone.BOARD),
            MinionInstance("overseer", "15.6:17", MinionZone.HAND),
        ),
    )
    engine.state = replace(engine.state, players=(custom,) + engine.state.players[1:])
    client.update(16)
    client.draw()

    def click(pos) -> None:
        event = pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=pos)
        client.handle_events([event])
        client.draw()

    click(_row_rects(1, HAND_Y, CARD_W, CARD_H)[0].center)
    assert client.targeting_id == "overseer"
    assert not engine.action_history

    board_rects = _row_rects(2, BOARD_Y, MINION_W, MINION_H, gap=12)
    click(board_rects[1].center)

    assert client.targeting_id is None
    record = engine.action_history[-1]
    assert (record.action.target_id, record.action.effect_target_id) == ("overseer", "demon-2")
    board = {m.instance_id: m for m in engine.player_state("player-1").board}
    assert board["demon-2"].attack_bonus == 2 and board["demon-1"].attack_bonus == 0
    assert "targeting Voidwalker" in client.log[-1][0]

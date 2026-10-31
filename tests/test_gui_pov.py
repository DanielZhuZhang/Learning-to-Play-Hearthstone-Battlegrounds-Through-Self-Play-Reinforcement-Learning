from hsbg.gui import visible_lines

LINES = [
    ("welcome", None),
    ("p1 bought", frozenset({"player-1"})),
    ("p2 bought", frozenset({"player-2"})),
    ("p1 vs p3 combat", frozenset({"player-1", "player-3"})),
    ("p2 vs ghost combat", frozenset({"player-2", "ghost:1"})),
    ("replay notice", None),
]


def test_no_pov_shows_every_line_in_order() -> None:
    assert visible_lines(LINES, None) == [text for text, _players in LINES]


def test_pov_shows_global_own_and_own_combat_lines_in_order() -> None:
    assert visible_lines(LINES, "player-1") == [
        "welcome",
        "p1 bought",
        "p1 vs p3 combat",
        "replay notice",
    ]


def test_pov_includes_combats_where_player_is_the_opponent() -> None:
    assert "p1 vs p3 combat" in visible_lines(LINES, "player-3")
    assert "p2 vs ghost combat" in visible_lines(LINES, "player-2")
    assert "p2 bought" not in visible_lines(LINES, "player-3")

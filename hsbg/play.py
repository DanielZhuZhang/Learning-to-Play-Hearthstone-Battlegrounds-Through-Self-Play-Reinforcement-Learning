"""Command-line entry point for the optional desktop client."""

from __future__ import annotations

import argparse
from pathlib import Path

from hsbg.data.profiles import load_profile
from hsbg.data.selection import apply_profile_overrides, select_enabled_cards
from hsbg.data.workbook import load_sheet

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Play against random bots or watch a bot-only match."
    )
    parser.add_argument(
        "--profile",
        type=Path,
        default=PROJECT_ROOT / "data" / "profiles" / "initial-15.6.json",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seat", default="player-1")
    parser.add_argument(
        "--spectate",
        action="store_true",
        help="Watch an all-bot match instead of controlling a player.",
    )
    parser.add_argument(
        "--pov",
        default=None,
        help="Start the GUI showing only this player's view (e.g. player-3).",
    )
    parser.add_argument(
        "--ui",
        choices=("pygame", "tk"),
        default="pygame",
        help="pygame draws a game-like table with animated combat; tk is the text client.",
    )
    args = parser.parse_args()

    profile = load_profile(args.profile)
    cards = apply_profile_overrides(
        profile,
        select_enabled_cards(
            profile,
            load_sheet(PROJECT_ROOT / profile.workbook, profile.sheet),
        ),
    )
    seats = {f"player-{index + 1}" for index in range(profile.rules.player_count)}
    if not args.spectate and args.seat not in seats:
        parser.error(f"Seat {args.seat!r} is not configured in this lobby.")
    if args.pov is not None and args.pov not in seats:
        parser.error(f"POV player {args.pov!r} is not configured in this lobby.")
    if args.ui == "pygame":
        from hsbg.pygame_gui import launch_pygame_gui as launch_gui
    else:
        from hsbg.gui import launch_gui
    launch_gui(
        profile,
        cards,
        args.seed,
        human_player_id=None if args.spectate else args.seat,
        pov_player_id=args.pov,
    )


if __name__ == "__main__":
    main()

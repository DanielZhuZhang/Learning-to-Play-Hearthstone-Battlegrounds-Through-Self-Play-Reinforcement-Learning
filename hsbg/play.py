"""Command-line entry point for the optional desktop client."""

from __future__ import annotations

import argparse
from pathlib import Path

from hsbg.data.profiles import load_profile
from hsbg.data.selection import apply_profile_overrides, select_enabled_cards
from hsbg.data.workbook import load_sheet
from hsbg.gui import launch_gui

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
    args = parser.parse_args()

    profile = load_profile(args.profile)
    cards = apply_profile_overrides(
        profile,
        select_enabled_cards(
            profile,
            load_sheet(PROJECT_ROOT / profile.workbook, profile.sheet),
        ),
    )
    if not args.spectate and args.seat not in {
        f"player-{index + 1}" for index in range(profile.rules.player_count)
    }:
        parser.error(f"Seat {args.seat!r} is not configured in this lobby.")
    launch_gui(
        profile,
        cards,
        args.seed,
        human_player_id=None if args.spectate else args.seat,
    )


if __name__ == "__main__":
    main()

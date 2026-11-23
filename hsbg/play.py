"""Command-line entry point for the optional desktop client."""

from __future__ import annotations

import argparse
from pathlib import Path

from hsbg.data.profiles import load_game_data

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
    parser.add_argument(
        "--agent",
        type=Path,
        default=None,
        help="PPO checkpoint (.pt) whose actor controls --agent-seats.",
    )
    parser.add_argument(
        "--agent-seats",
        default=None,
        help="Comma-separated seats for the trained agent "
        "(default: player-2 when playing, player-1 when spectating).",
    )
    parser.add_argument(
        "--agent-greedy",
        action="store_true",
        help="Always take the agent's top choice instead of sampling like in training.",
    )
    args = parser.parse_args()

    profile, cards = load_game_data(args.profile)
    seats = {f"player-{index + 1}" for index in range(profile.rules.player_count)}
    if not args.spectate and args.seat not in seats:
        parser.error(f"Seat {args.seat!r} is not configured in this lobby.")
    if args.pov is not None and args.pov not in seats:
        parser.error(f"POV player {args.pov!r} is not configured in this lobby.")
    human = None if args.spectate else args.seat

    agents = {}
    if args.agent is not None:
        if args.ui != "pygame":
            parser.error("--agent is only supported with --ui pygame.")
        if not args.agent.is_file():
            parser.error(f"Checkpoint not found: {args.agent}")
        default_seat = "player-1" if args.spectate else "player-2"
        agent_seats = [s.strip() for s in (args.agent_seats or default_seat).split(",") if s.strip()]
        for seat in agent_seats:
            if seat not in seats:
                parser.error(f"Agent seat {seat!r} is not configured in this lobby.")
            if seat == human:
                parser.error(f"Agent seat {seat!r} is your own seat.")
        from hsbg.agent_bot import TrainedBot

        bot = TrainedBot(args.agent, profile, cards, deterministic=args.agent_greedy)
        agents = {seat: bot for seat in agent_seats}
        print(f"Agent from {args.agent} (iteration {bot.iteration}) controls {', '.join(agent_seats)}")

    if args.ui == "pygame":
        from hsbg.pygame_gui import launch_pygame_gui

        launch_pygame_gui(
            profile, cards, args.seed,
            human_player_id=human, pov_player_id=args.pov, agents=agents,
        )
    else:
        from hsbg.gui import launch_gui

        launch_gui(profile, cards, args.seed, human_player_id=human, pov_player_id=args.pov)


if __name__ == "__main__":
    main()

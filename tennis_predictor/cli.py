"""Command-line entry point for reproducible tournament forecasts."""

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from . import __version__
from .io import load_players, write_results
from .models import Surface, TournamentConfig, TournamentLevel
from .reporting import format_projection, format_standings, save_probability_chart
from .simulation import project_bracket, run_simulations


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tennis-predictor",
        description="Estimate tennis tournament outcomes with Monte Carlo simulation.",
    )
    parser.add_argument("--players", type=Path, required=True, help="player CSV file")
    parser.add_argument(
        "--surface",
        choices=[surface.value for surface in Surface],
        default=Surface.HARD.value,
    )
    parser.add_argument(
        "--tournament",
        choices=[level.value for level in TournamentLevel],
        default=TournamentLevel.OPEN.value,
    )
    parser.add_argument("--simulations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--shuffle-draw",
        action="store_true",
        help="randomize the bracket before every simulation",
    )
    parser.add_argument("--output", type=Path, help="write detailed results to CSV")
    parser.add_argument("--chart", type=Path, help="save a probability chart")
    parser.add_argument(
        "--show-projection",
        action="store_true",
        help="print the highest-probability path through the fixed draw",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        players = load_players(args.players)
        config = TournamentConfig(
            surface=Surface(args.surface),
            level=TournamentLevel(args.tournament),
            simulations=args.simulations,
            seed=args.seed,
            shuffle_draw=args.shuffle_draw,
        )
        result = run_simulations(players, config)
        print(format_standings(result))
        print(f"\nSeed: {result.seed}")
        print("Model: transparent heuristic; not a betting recommendation.")

        if args.show_projection:
            if args.shuffle_draw:
                print(
                    "\nProjection uses the CSV order even though simulations "
                    "shuffle the draw."
                )
            print("\nProjected bracket")
            print(format_projection(project_bracket(players, config)))

        if args.output:
            write_results(args.output, result)
            print(f"\nWrote results to {args.output}")
        if args.chart:
            save_probability_chart(args.chart, result)
            print(f"Saved chart to {args.chart}")
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

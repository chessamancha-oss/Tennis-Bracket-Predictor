"""CSV input and output helpers."""

import csv
from pathlib import Path
from typing import List

from .models import Handedness, Player
from .simulation import TournamentResult, validate_bracket

PLAYER_COLUMNS = (
    "name",
    "handedness",
    "serve_accuracy",
    "return_accuracy",
    "aces_per_match",
    "double_faults_per_match",
    "recent_win_ratio",
    "straight_sets_win_ratio",
    "win_vs_right",
    "win_vs_left",
    "injury_impact",
)


def load_players(path: Path) -> List[Player]:
    try:
        handle = path.open("r", encoding="utf-8-sig", newline="")
    except OSError as error:
        raise ValueError(f"could not read player file '{path}': {error}") from error

    with handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("player CSV is missing a header row")

        missing = [
            column for column in PLAYER_COLUMNS if column not in reader.fieldnames
        ]
        if missing:
            raise ValueError("player CSV is missing columns: " + ", ".join(missing))

        players: List[Player] = []
        for row_number, row in enumerate(reader, start=2):
            if not any((value or "").strip() for value in row.values()):
                continue
            try:
                players.append(
                    Player(
                        name=row["name"],
                        handedness=Handedness(row["handedness"].strip().lower()),
                        serve_accuracy=float(row["serve_accuracy"]),
                        return_accuracy=float(row["return_accuracy"]),
                        aces_per_match=float(row["aces_per_match"]),
                        double_faults_per_match=float(row["double_faults_per_match"]),
                        recent_win_ratio=float(row["recent_win_ratio"]),
                        straight_sets_win_ratio=float(row["straight_sets_win_ratio"]),
                        win_vs_right=float(row["win_vs_right"]),
                        win_vs_left=float(row["win_vs_left"]),
                        injury_impact=float(row["injury_impact"]),
                    )
                )
            except (KeyError, TypeError, ValueError) as error:
                message = f"invalid player data on CSV row {row_number}: {error}"
                raise ValueError(message) from error

    validate_bracket(players)
    return players


def write_results(path: Path, result: TournamentResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "rank",
                "player",
                "championships",
                "probability",
                "confidence_95_low",
                "confidence_95_high",
            ]
        )
        for standing in result.standings:
            writer.writerow(
                [
                    standing.rank,
                    standing.player,
                    standing.championships,
                    f"{standing.probability:.8f}",
                    f"{standing.confidence_low:.8f}",
                    f"{standing.confidence_high:.8f}",
                ]
            )

#!/usr/bin/env python3
"""Build the Baseline Labs historical player catalogue and D1 seed migration.

The input is the Jeff Sackmann ATP/WTA archive (CC BY-NC-SA 4.0). The output
contains derived, career-level aggregates only and is intended for
non-commercial research use. See web/data/NOTICE.md.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

SURFACES = ("Hard", "Clay", "Grass")


def number(row: dict[str, str], key: str) -> int | None:
    value = row.get(key, "").strip()
    try:
        return int(float(value)) if value else None
    except ValueError:
        return None


def expected(first: float, second: float) -> float:
    return 1.0 / (1.0 + 10 ** ((second - first) / 400.0))


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def normalized(value: str) -> str:
    plain = unicodedata.normalize("NFKD", value)
    ascii_value = "".join(
        character for character in plain if not unicodedata.combining(character)
    )
    return re.sub(r"[^a-z0-9]+", " ", ascii_value.lower()).strip()


@dataclass
class Player:
    source_id: str
    tour: str
    name: str = ""
    country: str = "—"
    hand: str = "Unknown"
    birth_year: int | None = None
    first_year: int = 9999
    last_year: int = 0
    last_match_date: str | None = None
    matches: int = 0
    wins: int = 0
    rating: float = 1500.0
    peak_rating: float = 1500.0
    surface_rating: dict[str, float] = field(
        default_factory=lambda: defaultdict(lambda: 1500.0)
    )
    surface_peak: dict[str, float] = field(
        default_factory=lambda: defaultdict(lambda: 1500.0)
    )
    surface_matches: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    serve_won: int = 0
    serve_total: int = 0
    return_won: int = 0
    return_total: int = 0
    service_games: int = 0
    breaks_conceded: int = 0
    aces: int = 0
    double_faults: int = 0
    major_titles: int = 0
    rank: int | None = None
    ranking_points: int | None = None


def load_player_table(folder: Path, tour: str) -> dict[str, dict[str, str]]:
    path = folder / f"{tour.lower()}_players.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["player_id"]: row for row in csv.DictReader(handle)}


def load_latest_rankings(folder: Path, tour: str) -> dict[str, tuple[int, int | None]]:
    path = folder / f"{tour.lower()}_rankings_current.csv"
    if not path.exists():
        return {}
    latest: dict[str, tuple[str, int, int | None]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            player_id = row.get("player", "")
            when = row.get("ranking_date", "")
            rank = number(row, "rank")
            if not player_id or rank is None:
                continue
            current = latest.get(player_id)
            if current is None or when >= current[0]:
                latest[player_id] = (when, rank, number(row, "points"))
    return {
        player_id: (rank, points) for player_id, (_, rank, points) in latest.items()
    }


def match_files(folder: Path, tour: str) -> list[Path]:
    pattern = re.compile(rf"{tour.lower()}_matches_(\d{{4}})\.csv$")
    return sorted(
        path
        for path in folder.glob(f"{tour.lower()}_matches_*.csv")
        if pattern.search(path.name)
    )


def identity(
    player: Player, row: dict[str, str], prefix: str, source: dict[str, dict[str, str]]
) -> None:
    source_row = source.get(player.source_id, {})
    first = source_row.get("name_first", "").strip()
    last = source_row.get("name_last", "").strip()
    row_name = row.get("winner_name" if prefix == "w" else "loser_name", "").strip()
    player.name = (
        " ".join(part for part in (first, last) if part).strip()
        or row_name
        or player.name
    )
    ioc = (
        source_row.get("ioc", "").strip()
        or row.get("winner_ioc" if prefix == "w" else "loser_ioc", "").strip()
    )
    if ioc:
        player.country = ioc
    hand = (
        source_row.get("hand", "").strip()
        or row.get("winner_hand" if prefix == "w" else "loser_hand", "").strip()
    )
    if hand:
        player.hand = "Left" if hand == "L" else "Right" if hand == "R" else "Unknown"
    dob = source_row.get("dob", "").strip()
    if len(dob) >= 4 and dob[:4].isdigit():
        player.birth_year = int(dob[:4])


def add_point_stats(
    player: Player, row: dict[str, str], prefix: str, opponent_prefix: str
) -> None:
    serve_total = number(row, f"{prefix}_svpt")
    first_won = number(row, f"{prefix}_1stWon")
    second_won = number(row, f"{prefix}_2ndWon")
    opponent_total = number(row, f"{opponent_prefix}_svpt")
    opponent_first = number(row, f"{opponent_prefix}_1stWon")
    opponent_second = number(row, f"{opponent_prefix}_2ndWon")
    if None not in (serve_total, first_won, second_won):
        player.serve_total += serve_total or 0
        player.serve_won += (first_won or 0) + (second_won or 0)
        player.aces += number(row, f"{prefix}_ace") or 0
        player.double_faults += number(row, f"{prefix}_df") or 0
        player.service_games += number(row, f"{prefix}_SvGms") or 0
        player.breaks_conceded += max(
            0,
            (number(row, f"{prefix}_bpFaced") or 0)
            - (number(row, f"{prefix}_bpSaved") or 0),
        )
    if None not in (opponent_total, opponent_first, opponent_second):
        player.return_total += opponent_total or 0
        player.return_won += (
            (opponent_total or 0) - (opponent_first or 0) - (opponent_second or 0)
        )


def build_tour(folder: Path, tour: str) -> list[Player]:
    source = load_player_table(folder, tour)
    rankings = load_latest_rankings(folder, tour)
    players: dict[str, Player] = {}

    def get(source_id: str) -> Player:
        if source_id not in players:
            players[source_id] = Player(source_id=source_id, tour=tour)
        return players[source_id]

    for path in match_files(folder, tour):
        with path.open(newline="", encoding="utf-8") as handle:
            rows = sorted(
                csv.DictReader(handle),
                key=lambda row: (row.get("tourney_date", ""), row.get("match_num", "")),
            )
        for row in rows:
            winner_id = row.get("winner_id", "").strip()
            loser_id = row.get("loser_id", "").strip()
            if not winner_id or not loser_id:
                continue
            when = row.get("tourney_date", "")
            if len(when) < 4 or not when[:4].isdigit():
                continue
            year = int(when[:4])
            winner, loser = get(winner_id), get(loser_id)
            identity(winner, row, "w", source)
            identity(loser, row, "l", source)
            probability = expected(winner.rating, loser.rating)
            experience = min(winner.matches, loser.matches)
            k = 34.0 / (1.0 + experience / 140.0) ** 0.24
            shift = k * (1.0 - probability)
            winner.rating += shift
            loser.rating -= shift
            winner.peak_rating = max(winner.peak_rating, winner.rating)
            loser.peak_rating = max(loser.peak_rating, loser.rating)

            surface = row.get("surface", "")
            if surface in SURFACES:
                surface_probability = expected(
                    winner.surface_rating[surface], loser.surface_rating[surface]
                )
                surface_shift = (k + 4.0) * (1.0 - surface_probability)
                winner.surface_rating[surface] += surface_shift
                loser.surface_rating[surface] -= surface_shift
                winner.surface_peak[surface] = max(
                    winner.surface_peak[surface], winner.surface_rating[surface]
                )
                loser.surface_peak[surface] = max(
                    loser.surface_peak[surface], loser.surface_rating[surface]
                )
                winner.surface_matches[surface] += 1
                loser.surface_matches[surface] += 1

            for player, won, prefix, opponent_prefix in (
                (winner, True, "w", "l"),
                (loser, False, "l", "w"),
            ):
                player.matches += 1
                player.wins += int(won)
                player.first_year = min(player.first_year, year)
                player.last_year = max(player.last_year, year)
                try:
                    player.last_match_date = (
                        datetime.strptime(when, "%Y%m%d").date().isoformat()
                    )
                except ValueError:
                    pass
                add_point_stats(player, row, prefix, opponent_prefix)
            if row.get("tourney_level") == "G" and row.get("round") == "F":
                winner.major_titles += 1

    for source_id, player in players.items():
        if source_id in rankings and player.last_year >= 2025:
            player.rank, player.ranking_points = rankings[source_id]
    return [
        player for player in players.values() if player.matches >= 5 and player.name
    ]


def sql_value(value: object) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, (int, float)):
        return str(value)
    return "'" + str(value).replace("'", "''") + "'"


def record(player: Player) -> list[object]:
    active = player.last_year >= 2025
    overall = player.rating if active else player.peak_rating
    win_rate = player.wins / player.matches
    base_serve = 0.635 if player.tour == "ATP" else 0.585
    serve = (
        player.serve_won / player.serve_total
        if player.serve_total
        else clamp(base_serve + (overall - 1500.0) / 7000.0, 0.52, 0.72)
    )
    base_return = 1.0 - base_serve
    return_rate = (
        player.return_won / player.return_total
        if player.return_total
        else clamp(base_return + (overall - 1500.0) / 9000.0, 0.28, 0.49)
    )
    hold = (
        1.0 - player.breaks_conceded / player.service_games
        if player.service_games
        else clamp(0.75 + (serve - base_serve) * 2.1, 0.48, 0.94)
    )

    surface_values: list[object] = []
    for surface in SURFACES:
        count = player.surface_matches[surface]
        raw = player.surface_rating[surface] if active else player.surface_peak[surface]
        shrink = count / (count + 14.0)
        surface_values.extend(
            (round(shrink * raw + (1.0 - shrink) * overall, 1), count)
        )

    return [
        f"{player.tour.lower()}-{player.source_id}",
        player.tour,
        player.name,
        normalized(player.name),
        player.country,
        player.hand,
        player.birth_year,
        player.first_year,
        player.last_year,
        player.rank,
        player.ranking_points,
        round(overall, 1),
        round(
            clamp(
                56.0
                + 165.0 / math.sqrt(player.matches + 1.0)
                + (12 if player.serve_total == 0 else 0),
                58.0,
                170.0,
            ),
            1,
        ),
        player.matches,
        player.wins,
        round(clamp(win_rate, 0.05, 0.95), 4),
        round(serve, 4),
        round(return_rate, 4),
        round(clamp(hold, 0.35, 0.98), 4),
        round(player.aces / player.serve_total, 4) if player.serve_total else None,
        round(player.double_faults / player.serve_total, 4)
        if player.serve_total
        else None,
        player.serve_total,
        player.return_total,
        *surface_values,
        player.major_titles,
        player.last_match_date,
    ]


COLUMNS = [
    "id",
    "tour",
    "name",
    "search_key",
    "country",
    "hand",
    "birth_year",
    "career_start",
    "career_end",
    "rank",
    "ranking_points",
    "rating",
    "rating_sigma",
    "matches",
    "wins",
    "form_rate",
    "serve_points_won",
    "return_points_won",
    "hold_rate",
    "ace_rate",
    "double_fault_rate",
    "serve_sample",
    "return_sample",
    "hard_rating",
    "hard_matches",
    "clay_rating",
    "clay_matches",
    "grass_rating",
    "grass_matches",
    "major_titles",
    "last_match_date",
]


def write_migration(players: list[Player], output: Path) -> None:
    rows = sorted(
        (record(player) for player in players),
        key=lambda row: (str(row[1]), -float(row[11]), str(row[2])),
    )
    statements = ["DELETE FROM players"]
    batch_size = 35
    for start in range(0, len(rows), batch_size):
        values = rows[start : start + batch_size]
        tuples = [
            "(" + ",".join(sql_value(value) for value in row) + ")" for row in values
        ]
        statements.append(
            f"INSERT INTO players ({','.join(COLUMNS)}) VALUES\n" + ",\n".join(tuples)
        )
    statements.append("PRAGMA optimize")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        ";\n--> statement-breakpoint\n".join(statements) + ";\n", encoding="utf-8"
    )


def write_summary(players: list[Player], output: Path) -> None:
    starts = [player.first_year for player in players]
    payload = {
        "count": len(players),
        "atp": sum(player.tour == "ATP" for player in players),
        "wta": sum(player.tour == "WTA" for player in players),
        "firstYear": min(starts),
        "lastYear": max(player.last_year for player in players),
        "generatedAt": datetime.now().date().isoformat(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = json.dumps(payload, indent=2)
    output.write_text(
        "// Generated by scripts/build_player_database.py.\n"
        f"export const playerDatabaseSummary = {summary} as const;\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument(
        "--migration", type=Path, default=Path("web/drizzle/0001_seed_players.sql")
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("web/data/player-database-summary.generated.ts"),
    )
    args = parser.parse_args()
    players = build_tour(args.archive / "atp", "ATP") + build_tour(
        args.archive / "wta", "WTA"
    )
    write_migration(players, args.migration)
    write_summary(players, args.summary)
    print(f"Wrote {len(players):,} historical profiles")


if __name__ == "__main__":
    main()

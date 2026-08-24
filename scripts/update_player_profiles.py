#!/usr/bin/env python3
"""Build versioned professional-player priors from public match history.

The generated browser dataset contains derived aggregates only. Match history is
read from the Jeff Sackmann ATP/WTA datasets (CC BY-NC-SA 4.0); see
web/data/NOTICE.md for attribution and usage terms.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

DATA_CUTOFF = date(2026, 5, 25)
RANKING_SNAPSHOT = date(2026, 8, 18)

ROSTERS = {
    "ATP": [
        ("Jannik Sinner", "Jannik Sinner", 1, 14750),
        ("Carlos Alcaraz", "Carlos Alcaraz", 2, 11960),
        ("Alexander Zverev", "Alexander Zverev", 3, 5705),
        ("Novak Djokovic", "Novak Djokovic", 4, 4460),
        ("Ben Shelton", "Ben Shelton", 5, 4070),
        ("Felix Auger-Aliassime", "Felix Auger Aliassime", 6, 4050),
        ("Alex de Minaur", "Alex de Minaur", 7, 3855),
        ("Daniil Medvedev", "Daniil Medvedev", 8, 3760),
        ("Taylor Fritz", "Taylor Fritz", 9, 3720),
        ("Alexander Bublik", "Alexander Bublik", 10, 3320),
        ("Lorenzo Musetti", "Lorenzo Musetti", 11, 3115),
        ("Jiri Lehecka", "Jiri Lehecka", 12, 2665),
        ("Andrey Rublev", "Andrey Rublev", 13, 2460),
        ("Flavio Cobolli", "Flavio Cobolli", 14, 2340),
        ("Karen Khachanov", "Karen Khachanov", 15, 2320),
        ("Casper Ruud", "Casper Ruud", 16, 2275),
    ],
    "WTA": [
        ("Aryna Sabalenka", "Aryna Sabalenka", 1, 8670),
        ("Elena Rybakina", "Elena Rybakina", 2, 8316),
        ("Jessica Pegula", "Jessica Pegula", 3, 6680),
        ("Coco Gauff", "Coco Gauff", 4, 5919),
        ("Iga Swiatek", "Iga Swiatek", 5, 5419),
        ("Mirra Andreeva", "Mirra Andreeva", 6, 5323),
        ("Karolina Muchova", "Karolina Muchova", 7, 5048),
        ("Linda Noskova", "Linda Noskova", 8, 5016),
        ("Elina Svitolina", "Elina Svitolina", 9, 4634),
        ("Amanda Anisimova", "Amanda Anisimova", 10, 4353),
        ("Marta Kostyuk", "Marta Kostyuk", 11, 3830),
        ("Belinda Bencic", "Belinda Bencic", 12, 2995),
        ("Naomi Osaka", "Naomi Osaka", 13, 2846),
        ("Diana Shnaider", "Diana Shnaider", 14, 2798),
        ("Jasmine Paolini", "Jasmine Paolini", 15, 2773),
        ("Iva Jovic", "Iva Jovic", 16, 2691),
    ],
}


@dataclass
class Aggregate:
    wins: int = 0
    losses: int = 0
    last_match: date | None = None
    recent: list[tuple[date, bool]] = field(default_factory=list)
    surface_record: dict[str, list[int]] = field(
        default_factory=lambda: defaultdict(lambda: [0, 0])
    )
    serve_won: int = 0
    serve_total: int = 0
    return_won: int = 0
    return_total: int = 0
    aces: int = 0
    double_faults: int = 0
    service_games: int = 0
    breaks_conceded: int = 0
    hand: str = "U"
    country: str = "—"
    age: float | None = None


def number(row: dict[str, str], key: str) -> int | None:
    value = row.get(key, "").strip()
    try:
        return int(float(value)) if value else None
    except ValueError:
        return None


def expected(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def parse_matches(paths: Iterable[Path]):
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    return sorted(rows, key=lambda row: (row["tourney_date"], row["match_num"]))


def build_tour(archive: Path, tour: str) -> list[dict[str, object]]:
    folder = archive / tour.lower()
    rows = parse_matches(
        folder / f"{tour.lower()}_matches_{year}.csv" for year in (2024, 2025, 2026)
    )
    overall: dict[str, float] = defaultdict(lambda: 1500.0)
    surface_rating: dict[str, dict[str, float]] = defaultdict(
        lambda: defaultdict(lambda: 1500.0)
    )
    surface_count: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    aggregates: dict[str, Aggregate] = defaultdict(Aggregate)

    for row in rows:
        winner, loser = row["winner_name"], row["loser_name"]
        if not winner or not loser:
            continue
        match_date = datetime.strptime(row["tourney_date"], "%Y%m%d").date()
        surface = row.get("surface") or "Hard"
        k = 26.0
        p = expected(overall[winner], overall[loser])
        overall[winner] += k * (1.0 - p)
        overall[loser] += k * (0.0 - (1.0 - p))
        ps = expected(surface_rating[winner][surface], surface_rating[loser][surface])
        surface_rating[winner][surface] += 30.0 * (1.0 - ps)
        surface_rating[loser][surface] += 30.0 * (0.0 - (1.0 - ps))
        surface_count[winner][surface] += 1
        surface_count[loser][surface] += 1

        for name, won, prefix, opponent_prefix in (
            (winner, True, "w", "l"),
            (loser, False, "l", "w"),
        ):
            agg = aggregates[name]
            agg.wins += int(won)
            agg.losses += int(not won)
            agg.last_match = max(agg.last_match or match_date, match_date)
            agg.recent.append((match_date, won))
            agg.surface_record[surface][0 if won else 1] += 1
            agg.hand = (
                row.get(f"{prefix}inner_hand" if prefix == "w" else "loser_hand")
                or agg.hand
            )
            agg.country = (
                row.get(f"{prefix}inner_ioc" if prefix == "w" else "loser_ioc")
                or agg.country
            )
            age_key = "winner_age" if prefix == "w" else "loser_age"
            try:
                agg.age = float(row[age_key]) if row.get(age_key) else agg.age
            except ValueError:
                pass

            svpt = number(row, f"{prefix}_svpt")
            first_won = number(row, f"{prefix}_1stWon")
            second_won = number(row, f"{prefix}_2ndWon")
            opponent_svpt = number(row, f"{opponent_prefix}_svpt")
            opponent_first_won = number(row, f"{opponent_prefix}_1stWon")
            opponent_second_won = number(row, f"{opponent_prefix}_2ndWon")
            if None not in (svpt, first_won, second_won):
                agg.serve_total += svpt or 0
                agg.serve_won += (first_won or 0) + (second_won or 0)
                agg.aces += number(row, f"{prefix}_ace") or 0
                agg.double_faults += number(row, f"{prefix}_df") or 0
                agg.service_games += number(row, f"{prefix}_SvGms") or 0
                faced = number(row, f"{prefix}_bpFaced") or 0
                saved = number(row, f"{prefix}_bpSaved") or 0
                agg.breaks_conceded += max(0, faced - saved)
            if None not in (opponent_svpt, opponent_first_won, opponent_second_won):
                agg.return_total += opponent_svpt or 0
                agg.return_won += (
                    (opponent_svpt or 0)
                    - (opponent_first_won or 0)
                    - (opponent_second_won or 0)
                )

    profiles = []
    for display, source_name, rank, points in ROSTERS[tour]:
        agg = aggregates[source_name]
        recent = [
            (when, won) for when, won in agg.recent if (DATA_CUTOFF - when).days <= 365
        ]
        recent_90 = [
            (when, won) for when, won in recent if (DATA_CUTOFF - when).days <= 90
        ]
        total = agg.wins + agg.losses
        sample = len(recent)
        ranking_prior = 1500.0 + 400.0 * math.log10(max(points, 1) / 1000.0)
        history_weight = min(0.8, total / (total + 24.0))
        blended = (
            history_weight * overall[source_name]
            + (1.0 - history_weight) * ranking_prior
        )
        surfaces = {}
        surface_samples = {}
        for surface in ("Hard", "Clay", "Grass"):
            count = surface_count[source_name][surface]
            shrink = count / (count + 14.0)
            raw = surface_rating[source_name][surface]
            surfaces[surface.lower()] = round(
                shrink * raw + (1.0 - shrink) * blended, 1
            )
            surface_samples[surface.lower()] = count
        inactivity = (DATA_CUTOFF - agg.last_match).days if agg.last_match else 365
        sigma = min(
            190.0,
            52.0 + 145.0 / math.sqrt(sample + 1.0) + max(0, inactivity - 21) * 0.35,
        )

        serve_mean = agg.serve_won / agg.serve_total if agg.serve_total else 0.615
        return_mean = agg.return_won / agg.return_total if agg.return_total else 0.385
        hold_rate = (
            1.0 - agg.breaks_conceded / agg.service_games if agg.service_games else 0.78
        )
        recent_rate = sum(won for _, won in recent) / sample if sample else 0.5
        form_90 = (
            sum(won for _, won in recent_90) / len(recent_90)
            if recent_90
            else recent_rate
        )
        profiles.append(
            {
                "id": f"{tour.lower()}-{rank}",
                "tour": tour,
                "name": display,
                "rank": rank,
                "rankingPoints": points,
                "country": agg.country,
                "hand": "Left" if agg.hand == "L" else "Right",
                "age": round(agg.age, 1) if agg.age is not None else None,
                "rating": round(blended, 1),
                "surfaceRating": surfaces,
                "ratingSigma": round(sigma, 1),
                "matches52w": sample,
                "wins52w": sum(won for _, won in recent),
                "form90d": round(form_90, 4),
                "servePointsWon": round(serve_mean, 4),
                "returnPointsWon": round(return_mean, 4),
                "holdRate": round(max(0.45, min(0.98, hold_rate)), 4),
                "aceRate": round(agg.aces / agg.serve_total, 4)
                if agg.serve_total
                else None,
                "doubleFaultRate": round(agg.double_faults / agg.serve_total, 4)
                if agg.serve_total
                else None,
                "serveSample": agg.serve_total,
                "returnSample": agg.return_total,
                "surfaceSamples": surface_samples,
                "lastMatchDate": agg.last_match.isoformat() if agg.last_match else None,
                "rankingSnapshot": RANKING_SNAPSHOT.isoformat(),
                "historyCutoff": DATA_CUTOFF.isoformat(),
            }
        )
    return profiles


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--archive",
        type=Path,
        required=True,
        help="Path to the tennis-sackmann-archive checkout",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("web/data/players.generated.ts")
    )
    args = parser.parse_args()
    profiles = build_tour(args.archive, "ATP") + build_tour(args.archive, "WTA")
    payload = json.dumps(profiles, indent=2, ensure_ascii=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "// Generated by scripts/update_player_profiles.py. Do not edit by hand.\n"
        "// Rankings: ATP/WTA snapshot 2026-08-18. "
        "History: Sackmann archive through 2026-05-25.\n"
        f"export const professionalPlayers = {payload} as const;\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(profiles)} profiles to {args.output}")


if __name__ == "__main__":
    main()

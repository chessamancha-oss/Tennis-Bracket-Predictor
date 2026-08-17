"""Human-readable tables and optional chart output."""

from pathlib import Path
from typing import Sequence

from .simulation import MatchPrediction, TournamentResult


def format_standings(result: TournamentResult) -> str:
    name_width = max(6, *(len(item.player) for item in result.standings))
    header = (
        f"{'Rank':>4}  {'Player':<{name_width}}  {'Win chance':>10}  "
        f"{'95% interval':>17}  {'Titles':>7}"
    )
    separator = "-" * len(header)
    rows = [header, separator]
    for item in result.standings:
        interval = f"{item.confidence_low:.1%}–{item.confidence_high:.1%}"
        rows.append(
            f"{item.rank:>4}  {item.player:<{name_width}}  "
            f"{item.probability:>10.2%}  {interval:>17}  {item.championships:>7}"
        )
    return "\n".join(rows)


def format_projection(predictions: Sequence[MatchPrediction]) -> str:
    lines = []
    current_round = None
    for match in predictions:
        if match.round_number != current_round:
            current_round = match.round_number
            lines.append(f"Round {current_round}")
        lines.append(
            f"  {match.player_one} vs {match.player_two} -> "
            f"{match.projected_winner} ({match.winner_probability:.1%})"
        )
    return "\n".join(lines)


def save_probability_chart(path: Path, result: TournamentResult) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import PercentFormatter
    except ImportError as error:
        raise RuntimeError(
            "chart output requires the optional dependency: "
            "python -m pip install -e '.[charts]'"
        ) from error

    labels = [standing.player for standing in reversed(result.standings)]
    values = [standing.probability for standing in reversed(result.standings)]
    figure_height = max(4.0, len(labels) * 0.55)
    figure, axes = plt.subplots(figsize=(10, figure_height))
    bars = axes.barh(labels, values, color="#2563eb")
    axis_maximum = min(1.0, max(0.1, max(values) * 1.35))
    axes.set_xlim(0, axis_maximum)
    axes.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    axes.set_xlabel("Estimated championship probability")
    axes.set_title(f"Tournament forecast ({result.simulations:,} simulations)")
    axes.grid(axis="x", alpha=0.2)
    for bar, value in zip(bars, values):
        axes.text(
            min(value + axis_maximum * 0.012, axis_maximum * 0.96),
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1%}",
            va="center",
        )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)

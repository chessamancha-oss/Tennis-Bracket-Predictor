"""Transparent heuristic scoring and head-to-head probability calculations."""

import math
from dataclasses import dataclass

from .models import Handedness, Player, Surface, TournamentLevel


@dataclass(frozen=True)
class ModelWeights:
    serve: float = 1.15
    return_: float = 1.00
    aces: float = 0.20
    double_faults: float = 0.25
    recent_form: float = 0.90
    straight_sets: float = 0.35
    handedness_matchup: float = 0.55
    injury: float = 0.90
    logistic_scale: float = 1.75


DEFAULT_WEIGHTS = ModelWeights()
MIN_PROBABILITY = 0.02
MAX_PROBABILITY = 0.98


def _bounded_rate(value: float, reference_max: float) -> float:
    return min(value / reference_max, 1.0)


def player_score(
    player: Player,
    opponent: Player,
    surface: Surface,
    level: TournamentLevel,
    weights: ModelWeights = DEFAULT_WEIGHTS,
) -> float:
    """Return a transparent matchup score for one player.

    The score is meaningful only relative to the opponent's score under the
    same conditions. It is not itself a probability.
    """

    matchup_ratio = (
        player.win_vs_left
        if opponent.handedness is Handedness.LEFT
        else player.win_vs_right
    )
    ace_rate = _bounded_rate(player.aces_per_match, reference_max=20.0)
    double_fault_rate = _bounded_rate(
        player.double_faults_per_match, reference_max=10.0
    )

    score = (
        weights.serve * player.serve_accuracy
        + weights.return_ * player.return_accuracy
        + weights.aces * ace_rate
        - weights.double_faults * double_fault_rate
        + weights.recent_form * player.recent_win_ratio
        + weights.straight_sets * player.straight_sets_win_ratio
        + weights.handedness_matchup * matchup_ratio
        - weights.injury * player.injury_impact
    )

    if surface is Surface.GRASS:
        score += 0.30 * player.serve_accuracy + 0.20 * ace_rate
    elif surface is Surface.CLAY:
        score += 0.30 * player.return_accuracy + 0.12 * player.recent_win_ratio
    else:
        score += 0.14 * (player.serve_accuracy + player.return_accuracy)

    if level is TournamentLevel.GRAND_SLAM:
        score += 0.15 * player.straight_sets_win_ratio
        score -= 0.20 * player.injury_impact
    elif level is TournamentLevel.OPEN:
        score += 0.08 * player.recent_win_ratio

    return score


def match_win_probability(
    player: Player,
    opponent: Player,
    surface: Surface,
    level: TournamentLevel,
    weights: ModelWeights = DEFAULT_WEIGHTS,
) -> float:
    """Estimate a player's probability of beating an opponent.

    A logistic transform of the score difference guarantees complementary
    head-to-head probabilities before the defensive 2%/98% bounds.
    """

    difference = player_score(player, opponent, surface, level, weights) - player_score(
        opponent, player, surface, level, weights
    )
    raw_probability = 1.0 / (1.0 + math.exp(-weights.logistic_scale * difference))
    return min(max(raw_probability, MIN_PROBABILITY), MAX_PROBABILITY)

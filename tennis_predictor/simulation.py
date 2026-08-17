"""Single-elimination bracket projection and Monte Carlo simulation."""

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from .engine import DEFAULT_WEIGHTS, ModelWeights, match_win_probability
from .models import Player, TournamentConfig


@dataclass(frozen=True)
class Standing:
    rank: int
    player: str
    championships: int
    probability: float
    confidence_low: float
    confidence_high: float


@dataclass(frozen=True)
class MatchPrediction:
    round_number: int
    player_one: str
    player_two: str
    projected_winner: str
    winner_probability: float


@dataclass(frozen=True)
class TournamentResult:
    simulations: int
    seed: int
    standings: Tuple[Standing, ...]


def validate_bracket(players: Sequence[Player]) -> None:
    if len(players) < 2 or len(players) & (len(players) - 1):
        raise ValueError("player count must be a power of two and at least 2")

    normalized_names = [player.name.casefold() for player in players]
    if len(set(normalized_names)) != len(normalized_names):
        raise ValueError("player names must be unique")


def _simulate_match(
    player: Player,
    opponent: Player,
    config: TournamentConfig,
    rng: random.Random,
    weights: ModelWeights,
) -> Player:
    probability = match_win_probability(
        player, opponent, config.surface, config.level, weights
    )
    return player if rng.random() < probability else opponent


def _simulate_bracket(
    players: Sequence[Player],
    config: TournamentConfig,
    rng: random.Random,
    weights: ModelWeights,
) -> Player:
    current_round = list(players)
    if config.shuffle_draw:
        rng.shuffle(current_round)

    while len(current_round) > 1:
        current_round = [
            _simulate_match(
                current_round[index],
                current_round[index + 1],
                config,
                rng,
                weights,
            )
            for index in range(0, len(current_round), 2)
        ]
    return current_round[0]


def _wilson_interval(successes: int, trials: int) -> Tuple[float, float]:
    """Return a 95% Wilson score interval for a binomial proportion."""

    z = 1.959963984540054
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials * trials)
        )
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def run_simulations(
    players: Sequence[Player],
    config: TournamentConfig,
    weights: ModelWeights = DEFAULT_WEIGHTS,
) -> TournamentResult:
    validate_bracket(players)
    seed = (
        config.seed
        if config.seed is not None
        else random.SystemRandom().randrange(2**63)
    )
    rng = random.Random(seed)
    championships: Dict[str, int] = {player.name: 0 for player in players}

    for _ in range(config.simulations):
        champion = _simulate_bracket(players, config, rng, weights)
        championships[champion.name] += 1

    ordered = sorted(
        championships.items(), key=lambda item: (-item[1], item[0].casefold())
    )
    standings: List[Standing] = []
    for rank, (name, wins) in enumerate(ordered, start=1):
        low, high = _wilson_interval(wins, config.simulations)
        standings.append(
            Standing(
                rank=rank,
                player=name,
                championships=wins,
                probability=wins / config.simulations,
                confidence_low=low,
                confidence_high=high,
            )
        )

    return TournamentResult(
        simulations=config.simulations,
        seed=seed,
        standings=tuple(standings),
    )


def project_bracket(
    players: Sequence[Player],
    config: TournamentConfig,
    weights: ModelWeights = DEFAULT_WEIGHTS,
) -> Tuple[MatchPrediction, ...]:
    """Project the most likely winner of every match in the fixed input draw."""

    validate_bracket(players)
    current_round = list(players)
    predictions: List[MatchPrediction] = []
    round_number = 1

    while len(current_round) > 1:
        next_round: List[Player] = []
        for index in range(0, len(current_round), 2):
            player = current_round[index]
            opponent = current_round[index + 1]
            probability = match_win_probability(
                player, opponent, config.surface, config.level, weights
            )
            if probability >= 0.5:
                winner = player
                winner_probability = probability
            else:
                winner = opponent
                winner_probability = 1.0 - probability
            predictions.append(
                MatchPrediction(
                    round_number=round_number,
                    player_one=player.name,
                    player_two=opponent.name,
                    projected_winner=winner.name,
                    winner_probability=winner_probability,
                )
            )
            next_round.append(winner)
        current_round = next_round
        round_number += 1

    return tuple(predictions)

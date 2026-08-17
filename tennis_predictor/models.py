"""Validated domain models used by the prediction engine."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Surface(str, Enum):
    HARD = "hard"
    CLAY = "clay"
    GRASS = "grass"


class TournamentLevel(str, Enum):
    LOCAL = "local"
    OPEN = "open"
    GRAND_SLAM = "grand_slam"


class Handedness(str, Enum):
    RIGHT = "right"
    LEFT = "left"


@dataclass(frozen=True)
class Player:
    """A player's normalized performance profile.

    Accuracy, ratio, and impact values use the inclusive range 0..1. Ace and
    double-fault values are non-negative per-match averages.
    """

    name: str
    handedness: Handedness
    serve_accuracy: float
    return_accuracy: float
    aces_per_match: float
    double_faults_per_match: float
    recent_win_ratio: float
    straight_sets_win_ratio: float
    win_vs_right: float
    win_vs_left: float
    injury_impact: float

    def __post_init__(self) -> None:
        clean_name = self.name.strip()
        if not clean_name:
            raise ValueError("player name cannot be empty")
        object.__setattr__(self, "name", clean_name)

        if not isinstance(self.handedness, Handedness):
            raise ValueError("handedness must be 'right' or 'left'")

        ratio_fields = (
            "serve_accuracy",
            "return_accuracy",
            "recent_win_ratio",
            "straight_sets_win_ratio",
            "win_vs_right",
            "win_vs_left",
            "injury_impact",
        )
        for field_name in ratio_fields:
            value = getattr(self, field_name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be between 0 and 1")

        for field_name in ("aces_per_match", "double_faults_per_match"):
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} cannot be negative")


@dataclass(frozen=True)
class TournamentConfig:
    surface: Surface
    level: TournamentLevel
    simulations: int = 10_000
    seed: Optional[int] = 42
    shuffle_draw: bool = False

    def __post_init__(self) -> None:
        if self.simulations <= 0:
            raise ValueError("simulations must be greater than zero")
        if not isinstance(self.surface, Surface):
            raise ValueError("surface must be hard, clay, or grass")
        if not isinstance(self.level, TournamentLevel):
            raise ValueError("tournament level must be local, open, or grand_slam")

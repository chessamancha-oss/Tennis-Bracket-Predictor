import unittest

from tennis_predictor.models import Surface, TournamentConfig, TournamentLevel
from tennis_predictor.simulation import (
    project_bracket,
    run_simulations,
    validate_bracket,
)
from tests.helpers import make_player


class SimulationTests(unittest.TestCase):
    def setUp(self):
        self.players = [
            make_player("A", recent_win_ratio=0.90),
            make_player("B", recent_win_ratio=0.65),
            make_player("C", serve_accuracy=0.78),
            make_player("D", return_accuracy=0.75),
        ]
        self.config = TournamentConfig(
            surface=Surface.HARD,
            level=TournamentLevel.OPEN,
            simulations=2_000,
            seed=123,
        )

    def test_seeded_run_is_reproducible(self):
        first = run_simulations(self.players, self.config)
        second = run_simulations(self.players, self.config)
        self.assertEqual(first, second)

    def test_probabilities_sum_to_one(self):
        result = run_simulations(self.players, self.config)
        total = sum(standing.probability for standing in result.standings)
        self.assertAlmostEqual(total, 1.0)

    def test_confidence_intervals_contain_estimate(self):
        result = run_simulations(self.players, self.config)
        for standing in result.standings:
            self.assertLessEqual(standing.confidence_low, standing.probability)
            self.assertGreaterEqual(standing.confidence_high, standing.probability)

    def test_projection_contains_every_match(self):
        projection = project_bracket(self.players, self.config)
        self.assertEqual(len(projection), len(self.players) - 1)
        self.assertEqual(projection[-1].round_number, 2)

    def test_rejects_non_power_of_two_bracket(self):
        with self.assertRaisesRegex(ValueError, "power of two"):
            validate_bracket(self.players[:3])

    def test_rejects_duplicate_names_case_insensitively(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_bracket([make_player("A"), make_player("a")])


if __name__ == "__main__":
    unittest.main()

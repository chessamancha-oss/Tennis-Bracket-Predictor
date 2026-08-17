import unittest

from tennis_predictor.reporting import format_projection, format_standings
from tennis_predictor.simulation import MatchPrediction, Standing, TournamentResult


class ReportingTests(unittest.TestCase):
    def test_formats_ranked_standings_with_confidence_interval(self):
        result = TournamentResult(
            simulations=100,
            seed=42,
            standings=(
                Standing(
                    rank=1,
                    player="Avery",
                    championships=60,
                    probability=0.60,
                    confidence_low=0.50,
                    confidence_high=0.69,
                ),
            ),
        )
        table = format_standings(result)
        self.assertIn("Avery", table)
        self.assertIn("60.00%", table)
        self.assertIn("50.0%–69.0%", table)

    def test_formats_projection_grouped_by_round(self):
        predictions = (
            MatchPrediction(1, "A", "B", "A", 0.60),
            MatchPrediction(1, "C", "D", "D", 0.55),
            MatchPrediction(2, "A", "D", "A", 0.52),
        )
        output = format_projection(predictions)
        self.assertEqual(output.count("Round 1"), 1)
        self.assertEqual(output.count("Round 2"), 1)
        self.assertIn("A vs D -> A (52.0%)", output)


if __name__ == "__main__":
    unittest.main()

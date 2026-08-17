import unittest

from tennis_predictor.engine import match_win_probability
from tennis_predictor.models import Handedness, Surface, TournamentLevel
from tests.helpers import make_player


class MatchProbabilityTests(unittest.TestCase):
    def test_equal_profiles_produce_even_match(self):
        player = make_player("A")
        opponent = make_player("B")
        probability = match_win_probability(
            player, opponent, Surface.HARD, TournamentLevel.OPEN
        )
        self.assertAlmostEqual(probability, 0.5)

    def test_opposite_probabilities_are_complementary(self):
        player = make_player("A", serve_accuracy=0.82, recent_win_ratio=0.80)
        opponent = make_player("B", return_accuracy=0.77, recent_win_ratio=0.65)
        forward = match_win_probability(
            player, opponent, Surface.GRASS, TournamentLevel.GRAND_SLAM
        )
        reverse = match_win_probability(
            opponent, player, Surface.GRASS, TournamentLevel.GRAND_SLAM
        )
        self.assertAlmostEqual(forward + reverse, 1.0)

    def test_injury_reduces_win_probability(self):
        healthy = make_player("Healthy", injury_impact=0.0)
        injured = make_player("Injured", injury_impact=0.45)
        probability = match_win_probability(
            healthy, injured, Surface.HARD, TournamentLevel.GRAND_SLAM
        )
        self.assertGreater(probability, 0.5)

    def test_actual_opponent_handedness_is_used(self):
        specialist = make_player("Specialist", win_vs_left=0.90, win_vs_right=0.45)
        left_opponent = make_player("Left", handedness=Handedness.LEFT)
        right_opponent = make_player("Right", handedness=Handedness.RIGHT)
        against_left = match_win_probability(
            specialist, left_opponent, Surface.HARD, TournamentLevel.LOCAL
        )
        against_right = match_win_probability(
            specialist, right_opponent, Surface.HARD, TournamentLevel.LOCAL
        )
        self.assertGreater(against_left, against_right)

    def test_probabilities_are_defensively_bounded(self):
        favorite = make_player(
            "Favorite",
            serve_accuracy=1,
            return_accuracy=1,
            aces_per_match=30,
            recent_win_ratio=1,
            straight_sets_win_ratio=1,
            win_vs_right=1,
            injury_impact=0,
        )
        underdog = make_player(
            "Underdog",
            serve_accuracy=0,
            return_accuracy=0,
            aces_per_match=0,
            double_faults_per_match=20,
            recent_win_ratio=0,
            straight_sets_win_ratio=0,
            win_vs_right=0,
            injury_impact=1,
        )
        probability = match_win_probability(
            favorite, underdog, Surface.GRASS, TournamentLevel.GRAND_SLAM
        )
        self.assertEqual(probability, 0.98)


if __name__ == "__main__":
    unittest.main()

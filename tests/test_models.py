import unittest

from tennis_predictor.models import Surface, TournamentConfig, TournamentLevel
from tests.helpers import make_player


class PlayerValidationTests(unittest.TestCase):
    def test_strips_player_name(self):
        self.assertEqual(make_player("  Avery  ").name, "Avery")

    def test_rejects_empty_name(self):
        with self.assertRaisesRegex(ValueError, "name"):
            make_player("   ")

    def test_rejects_ratio_outside_unit_interval(self):
        with self.assertRaisesRegex(ValueError, "serve_accuracy"):
            make_player(serve_accuracy=1.01)

    def test_rejects_negative_counting_stat(self):
        with self.assertRaisesRegex(ValueError, "aces_per_match"):
            make_player(aces_per_match=-1)

    def test_rejects_untyped_handedness(self):
        with self.assertRaisesRegex(ValueError, "handedness"):
            make_player(handedness="right")


class ConfigValidationTests(unittest.TestCase):
    def test_rejects_nonpositive_simulation_count(self):
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            TournamentConfig(
                surface=Surface.HARD,
                level=TournamentLevel.OPEN,
                simulations=0,
            )

    def test_accepts_none_seed_for_nondeterministic_runs(self):
        config = TournamentConfig(
            surface=Surface.CLAY,
            level=TournamentLevel.LOCAL,
            seed=None,
        )
        self.assertIsNone(config.seed)


if __name__ == "__main__":
    unittest.main()

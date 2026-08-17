import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from tennis_predictor.cli import main
from tennis_predictor.io import load_players

VALID_CSV = (
    "name,handedness,serve_accuracy,return_accuracy,aces_per_match,"
    "double_faults_per_match,recent_win_ratio,straight_sets_win_ratio,"
    "win_vs_right,win_vs_left,injury_impact\n"
    "A,right,0.7,0.7,5,2,0.7,0.6,0.7,0.65,0.1\n"
    "B,left,0.6,0.8,3,1,0.6,0.5,0.6,0.7,0.0\n"
)


class CsvAndCliTests(unittest.TestCase):
    def test_loads_valid_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "players.csv"
            path.write_text(VALID_CSV, encoding="utf-8")
            players = load_players(path)
        self.assertEqual([player.name for player in players], ["A", "B"])

    def test_reports_missing_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "players.csv"
            path.write_text("name,handedness\nA,right\nB,left\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing columns"):
                load_players(path)

    def test_cli_writes_results(self):
        with tempfile.TemporaryDirectory() as directory:
            player_path = Path(directory) / "players.csv"
            result_path = Path(directory) / "forecast.csv"
            player_path.write_text(VALID_CSV, encoding="utf-8")
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = main(
                    [
                        "--players",
                        str(player_path),
                        "--simulations",
                        "200",
                        "--seed",
                        "7",
                        "--shuffle-draw",
                        "--show-projection",
                        "--output",
                        str(result_path),
                    ]
                )
            self.assertEqual(exit_code, 0)
            self.assertIn("Win chance", stdout.getvalue())
            self.assertIn("Projected bracket", stdout.getvalue())
            self.assertIn("simulations shuffle the draw", stdout.getvalue())
            self.assertIn("confidence_95_low", result_path.read_text(encoding="utf-8"))

    def test_cli_returns_error_for_bad_input(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            exit_code = main(["--players", "does-not-exist.csv"])
        self.assertEqual(exit_code, 2)
        self.assertIn("could not read", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()

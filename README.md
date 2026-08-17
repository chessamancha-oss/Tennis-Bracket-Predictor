# Tennis Bracket Predictor

A reproducible command-line tool for exploring single-elimination tennis tournament outcomes with Monte Carlo simulation.

The project turns player profiles and tournament conditions into head-to-head probabilities, simulates a full bracket thousands of times, and reports championship estimates with 95% confidence intervals. The scoring model is transparent and intentionally heuristic: every factor is visible in the source and can be tested or replaced.

> **Project status:** Alpha. The engineering workflow is production-shaped, but the model has not been calibrated against a historical match dataset. Its output is educational and experimental—not betting advice.

## What it does

- Reads a seeded bracket from a validated CSV file.
- Models serve, return, aces, double faults, recent form, straight-set form, handedness matchups, and injury impact.
- Applies surface-specific adjustments for hard, clay, and grass courts.
- Applies local, Open, or Grand Slam tournament context.
- Uses complementary head-to-head probabilities from a logistic score difference.
- Runs reproducible simulations using an explicit random seed.
- Reports estimated title probabilities and Wilson 95% confidence intervals.
- Exports machine-readable CSV results.
- Optionally saves a Matplotlib probability chart.
- Includes unit tests and a multi-version GitHub Actions workflow.

## Quick start

Python 3.9 or newer is required.

```bash
git clone https://github.com/chessamancha-oss/Tennis-Bracket-Predictor.git
cd Tennis-Bracket-Predictor
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Run the included eight-player example:

```bash
tennis-predictor \
  --players examples/players.csv \
  --surface hard \
  --tournament grand_slam \
  --simulations 20000 \
  --seed 42 \
  --show-projection \
  --output results/forecast.csv
```

The module form works without installing the console command:

```bash
python -m tennis_predictor --players examples/players.csv
```

To create charts, install the optional dependency and add `--chart`:

```bash
python -m pip install -e ".[charts]"
tennis-predictor --players examples/players.csv --chart results/forecast.png
```

## Input format

The CSV row order is the initial bracket order: row 1 plays row 2, row 3 plays row 4, and so on. The number of players must be a power of two and player names must be unique.

| Column | Type | Meaning |
| --- | --- | --- |
| `name` | text | Unique player name |
| `handedness` | `right` or `left` | Playing hand |
| `serve_accuracy` | 0–1 | Normalized serve performance |
| `return_accuracy` | 0–1 | Normalized return performance |
| `aces_per_match` | 0 or greater | Average aces per match |
| `double_faults_per_match` | 0 or greater | Average double faults per match |
| `recent_win_ratio` | 0–1 | Recent match win rate |
| `straight_sets_win_ratio` | 0–1 | Share of wins in straight sets |
| `win_vs_right` | 0–1 | Win rate against right-handed players |
| `win_vs_left` | 0–1 | Win rate against left-handed players |
| `injury_impact` | 0–1 | Estimated current injury impact; higher is worse |

Use `--shuffle-draw` to randomize the bracket before every simulation instead of treating CSV order as fixed.

## CLI reference

```text
--players PATH             Required player CSV
--surface VALUE            hard, clay, or grass (default: hard)
--tournament VALUE         local, open, or grand_slam (default: open)
--simulations N            Monte Carlo runs (default: 10000)
--seed N                   Random seed (default: 42)
--shuffle-draw             Randomize each simulated draw
--show-projection          Print the likely fixed-bracket path
--output PATH              Write detailed results as CSV
--chart PATH               Save a probability chart (optional dependency)
```

## Model design

For each potential matchup, the engine creates a score for both players from the entered profile, opponent handedness, surface, and tournament level. The difference between those scores is passed through a logistic function:

```text
P(A beats B) = 1 / (1 + exp(-scale × (score(A) - score(B))))
```

This makes the matchup internally consistent: before defensive bounds, `P(A beats B) + P(B beats A) = 1`. Probabilities are bounded at 2% and 98% to avoid false certainty from a small heuristic model.

Tournament probabilities are empirical championship frequencies across the requested simulations. The reported confidence interval describes Monte Carlo sampling uncertainty only; it does not measure model accuracy.

Default weights live in [`tennis_predictor/engine.py`](tennis_predictor/engine.py). Pass a custom `ModelWeights` object through the Python API to experiment without editing global state.

## Python API

```python
from pathlib import Path

from tennis_predictor.io import load_players
from tennis_predictor.models import Surface, TournamentConfig, TournamentLevel
from tennis_predictor.simulation import run_simulations

players = load_players(Path("examples/players.csv"))
config = TournamentConfig(
    surface=Surface.CLAY,
    level=TournamentLevel.OPEN,
    simulations=50_000,
    seed=2026,
)
result = run_simulations(players, config)

for standing in result.standings:
    print(standing.player, standing.probability)
```

## Development

Install the development tools, then run lint, formatting, and coverage checks:

```bash
python -m pip install -e ".[dev]"
ruff check .
ruff format --check .
coverage run -m unittest discover -s tests -v
coverage report --fail-under=85
```

Run a syntax and packaging smoke check:

```bash
python -m compileall -q tennis_predictor tests
python -m tennis_predictor --players examples/players.csv --simulations 1000
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the expected workflow.

## Known limitations and responsible use

- The weights are expert-style assumptions, not fitted coefficients.
- The model does not yet include rankings, Elo, surface-specific history, fatigue, travel, weather, best-of-three versus best-of-five format, or live injury data.
- Input quality directly controls output quality.
- Confidence intervals quantify simulation noise, not prediction validity.
- The example players are fictional and exist only to demonstrate the file format.

A serious forecasting model should train and backtest against time-split historical data, compare itself with simple baselines, measure calibration and log loss, and publish versioned evaluation results before making accuracy claims.

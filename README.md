# Tennis Bracket Predictor

An interactive Python simulator that estimates each player's chance of winning a single-elimination tennis tournament.

The program collects player statistics, calculates matchup probabilities, runs 5,000 Monte Carlo tournament simulations, and displays both a projected bracket and an overall win-probability chart.

## Features

- Supports any power-of-two bracket size, such as 8, 16, or 32 players.
- Uses serve accuracy, return accuracy, aces, double faults, recent form, straight-set results, handedness matchups, and injury impact.
- Adjusts player scores for grass, hard, and clay courts.
- Includes Open, Grand Slam, and local tournament contexts.
- Prints estimated championship probabilities for every player.
- Produces bracket and horizontal probability-chart visualizations.

## Requirements

- Python 3.10 or newer
- pandas
- NumPy
- Matplotlib

Install the Python packages with:

```bash
python -m pip install -r requirements.txt
```

## Run the simulator

From the repository directory, run:

```bash
python "# tennis_match_simulator.py"
```

Enter a power-of-two number of players and provide the requested statistics for each player. Ratio fields should be entered as decimal values between `0` and `1`.

Example:

```text
Serve accuracy: 0.68
Return accuracy: 0.42
Recent win ratio: 0.75
Recent injuries impact: 0.10
```

After collecting the players, the program asks for the court and tournament types, performs the simulations, prints the championship probabilities, and opens two Matplotlib charts.

## How the estimate works

Each matchup receives a weighted score based on the entered performance statistics. Court surface and tournament context modify that score. A random draw then selects the winner according to the resulting matchup probability. Repeating the full bracket 5,000 times produces an estimated title probability for every player.

This is a heuristic simulation—not a trained machine-learning model—and its output should be treated as an experiment rather than a betting or forecasting recommendation.

## Current limitations

- Player data must be entered manually each time.
- Inputs are not yet validated beyond the bracket-size assertion.
- Opponent handedness is approximated from age parity rather than collected directly.
- The displayed bracket chooses the higher-probability player at each step; it is a projection, not one of the simulated brackets.
- No automated tests are included yet.

## Suggested next steps

- Accept player data from CSV files.
- Collect handedness explicitly.
- Validate numeric ranges and provide clearer input errors.
- Make the random seed configurable for reproducible runs.
- Separate data collection, probability calculation, simulation, and visualization into testable modules.
- Add unit tests for matchup scoring and bracket advancement.

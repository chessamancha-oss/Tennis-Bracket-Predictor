# Contributing

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[charts,dev]"
```

## Before opening a pull request

1. Keep the prediction engine free of file, terminal, and plotting side effects.
2. Add or update tests for every behavior change.
3. Run `ruff check .` and `ruff format --check .`.
4. Run `coverage run -m unittest discover -s tests -v` and `coverage report`.
5. Run `python -m compileall -q tennis_predictor tests`.
6. Update the README when the CLI, input schema, or model assumptions change.

## Model changes

Changes to weights or probability calculations should explain the rationale and include evidence. Historical evaluation should use chronological train/test splits to avoid future-data leakage. Report calibration and a proper scoring rule such as log loss or Brier score, not only winner accuracy.

## Scope

Keep this repository focused on reproducible tennis matchup and tournament forecasting. Avoid presenting unvalidated estimates as betting guidance.

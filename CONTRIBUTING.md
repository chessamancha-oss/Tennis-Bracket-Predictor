# Contributing

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[charts,dev]"

cd web
pnpm install --frozen-lockfile
```

## Before opening a pull request

1. Keep the prediction engine free of file, terminal, and plotting side effects.
2. Add or update tests for every behavior change.
3. Run `ruff check .` and `ruff format --check .`.
4. Run `coverage run -m unittest discover -s tests -v` and `coverage report`.
5. Run `python -m compileall -q tennis_predictor tests`.
6. From `web/`, run `pnpm exec tsc --noEmit`, `pnpm lint`, and `pnpm test`.
7. Update the README and `MODEL_CARD.md` when the UI, input schema, data cutoff, or model assumptions change.

## Model changes

Changes to priors, posterior sampling, rating updates, or scoring simulation should explain the rationale and include evidence. Historical evaluation should use chronological train/test splits to avoid future-data leakage. Report calibration and a proper scoring rule such as log loss or Brier score, not only winner accuracy.

Generated professional profiles must be reproducible through `scripts/update_player_profiles.py`. Verify official rankings independently, preserve the historical-data attribution in `web/data/NOTICE.md`, and commit the generated diff with its exact snapshot dates.

## Scope

Keep this repository focused on reproducible tennis matchup and tournament forecasting. Avoid presenting unvalidated estimates as betting guidance.

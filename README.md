# Baseline Labs — Tennis Predictor

Baseline Labs is a professional tennis forecasting workspace. It combines a 7,255-player historical catalogue, surface-aware latent ability, Bayesian serve and return estimates, point-by-point Monte Carlo scoring, custom tournament simulation, a self-refreshing live tour desk, and a source-visible match-context layer.

The repository also retains a tested Python bracket CLI for reproducible batch experiments. The web product is the primary v3 experience.

> Research status: the architecture is probabilistic and the data pipeline is reproducible, but the model has not yet completed the rolling-origin calibration study described in the [model card](MODEL_CARD.md). It is not betting advice.

## Product experience

### 1v1 lab

- Search 3,513 ATP and 3,742 WTA players across 1967–2026.
- Compare current priors for active players or career-peak priors for retired players.
- Filter by tour and era, with name, country, career span, rating, and major-title context.
- Select hard, clay, or grass and best-of-three or best-of-five scoring.
- Run 5,040 complete match simulations across 36 posterior skill draws.
- Review win probability, an 80% model-uncertainty interval, likely set score, expected sets and games, tiebreak likelihood, and separate evidence layers.

### Unrestricted custom profiles

- Enter raw overall and surface ratings, rating uncertainty, serve/return rates, recent form, evidence volume, and signed pressure/endurance indices.
- Use decimals or percentages without a fixed 1–10 scale or UI ceiling.
- Read a dynamic interpretation for every entered value.
- Keep extreme inputs numerically stable through bounded probabilistic transforms rather than silently rejecting them.

### Context intelligence

- Select a current ATP or WTA event for a same-tour professional matchup.
- Read current or match-hour temperature, humidity, wind, precipitation, and elevation from the National Weather Service for U.S. venues, with Open-Meteo for global coverage and fallback.
- Estimate recovery time, recent venue-to-venue travel distance, and time-zone load from the preceding 21-day result window.
- Discover recent injury, withdrawal, recovery, and coaching-change reporting with direct publisher links.
- Apply only trusted or independently corroborated adverse availability reporting directionally; a lone unverified headline only widens uncertainty.
- Convert a confirmed selected-event withdrawal into a walkover instead of simulating a match that cannot occur.

### Bracket lab

- Add professionals from the historical catalogue, add custom entrants, or paste an entire roster.
- Use any field size; non-power-of-two draws receive automatic byes.
- Choose tournament category, surface, and best-of-three or best-of-five scoring.
- Inspect every projected matchup with win percentages, likely set scores, advancing players, and champion path.

### Live tour desk

- Reads the ATP and WTA tournaments active on the current date.
- Shows completed scores, in-progress match state, unresolved draw slots, and forecasts for known future matchups.
- Refreshes every 60 seconds so completed results lock into the bracket and newly resolved matchups receive new forecasts.
- Offers an on-demand contextual recalculation for each known live matchup.
- Currently uses the ESPN tennis scoreboard as the operational live-result layer; official tournament links remain available for verification.

## Model overview

This is not a weighted checklist. The v3 browser model is generative:

1. match history updates overall and surface-specific paired-comparison ratings;
2. sparse surface histories are partially pooled toward overall ability;
3. serve and return point skills are drawn from beta posteriors;
4. latent ability is drawn from uncertainty-aware normal posteriors;
5. open-ended pressure and fitness inputs act only in the match states they describe;
6. the simulator plays points into advantage games, tiebreaks, sets, and full matches; and
7. bounded context adjustments alter the latent player profiles before simulation; and
8. bracket forecasts repeat the same engine through a complete single-elimination path, preserving byes.

See [MODEL_CARD.md](MODEL_CARD.md) for assumptions, intended use, limitations, and the evaluation roadmap.

## Run the website

Node.js 22.13+ and pnpm 11 are required.

```bash
cd web
pnpm install --frozen-lockfile --ignore-scripts
pnpm dev
```

Then open `http://localhost:3000`.

Quality checks:

```bash
cd web
pnpm exec tsc --noEmit
pnpm lint
pnpm test
```

## Refresh professional player profiles

The generated player file is committed so every forecast is reproducible. To refresh it, check out the attributed archive and run:

```bash
git clone --depth 1 https://github.com/Aneeshers/tennis-sackmann-archive.git ../tennis-sackmann-archive
python scripts/update_player_profiles.py \
  --archive ../tennis-sackmann-archive \
  --output web/data/players.generated.ts
```

To rebuild the complete historical D1 catalogue and its migration:

```bash
python scripts/build_player_database.py \
  --archive ../tennis-sackmann-archive
```

Before publishing a refresh, independently verify current-ranking constants, update displayed cutoff dates, inspect the generated SQL and summary, run the full test suite, and review the generated diff. Historical and derived player data is CC BY-NC-SA 4.0; see [web/data/NOTICE.md](web/data/NOTICE.md).

## Continuous maintenance cadence

Production uses different cadences for data with different rates of change:

- the browser requests live draw state every 60 seconds;
- weather, travel, availability, injury, coaching, and news evidence is retrieved when a context forecast runs and is only cached briefly;
- an hourly maintenance heartbeat checks the production site, draw feed, context pipeline, source timestamps, and upstream response shapes;
- the first heartbeat each America/New_York calendar day checks the attributed player archive and official ATP/WTA ranking sources for a newer trustworthy snapshot, then regenerates and validates the catalogue only when one exists; and
- the first Monday heartbeat reviews production dependencies and security advisories, accepting only compatible low-risk changes that pass the complete quality gate.

Healthy feeds do not produce timestamp-only commits. A source or code change reaches production only after its generated diff is reviewed, the full test and build gate passes, and the exact passing commit is deployed.

## Python bracket CLI

The Python package remains useful for deterministic batch bracket experiments:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
tennis-predictor \
  --players examples/players.csv \
  --surface hard \
  --simulations 20000 \
  --seed 42 \
  --output results/forecast.csv
```

The CLI input schema and API are documented through `tennis-predictor --help` and the typed Python modules under `tennis_predictor/`.

## Repository map

```text
web/                         Interactive forecasting studio, D1 schema, and v3 model
scripts/update_player_profiles.py
                             Reproducible professional-profile generator
scripts/build_player_database.py
                             Historical catalogue and D1 seed generator
tennis_predictor/            Python bracket experimentation package
tests/                       Python unit and integration tests
.github/workflows/ci.yml     Python matrix + web type/lint/model/build checks
MODEL_CARD.md                Model behavior, data, risks, evaluation roadmap
```

## Data and research attribution

Current ranking snapshots are sourced from the official [ATP rankings](https://www.atptour.com/en/rankings/singles) and [WTA rankings](https://www.wtatennis.com/rankings/singles) pages. Historical aggregates are derived from datasets compiled by Jeff Sackmann / Tennis Abstract and used under CC BY-NC-SA 4.0.

The architecture is informed by paired-comparison research showing the value of time dynamics, surface covariates, and explicit uncertainty in tennis forecasting; see [Ingram, *Gaussian Process Priors for Dynamic Paired Comparison Modelling*](https://arxiv.org/abs/1902.07378).

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md). Changes to model behavior should include tests, model-card updates, and a note about calibration impact. Changes to generated player data must preserve attribution and cutoff provenance.

## License and responsible use

Source-code licensing and third-party data licensing are distinct. The historical and derived player data under `web/data/` carries CC BY-NC-SA 4.0 terms. Do not use the product as a source of guaranteed outcomes or as a substitute for current, verified player information.

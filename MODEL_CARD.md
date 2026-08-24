# Baseline Labs model card

## Intended use

Baseline Labs is research software for exploring pre-match tennis forecasts. It supports two inputs:

1. versioned ATP or WTA player profiles derived from official ranking snapshots and historical match results; and
2. custom scouting profiles whose 1–10 inputs define probability distributions and uncertainty, rather than a direct weighted score.

It is not validated for wagering, player health decisions, selection decisions, or any other high-stakes use.

## Architecture

The browser model has four layers:

- **Dynamic paired-comparison ability.** Sequential Elo-style ratings learn from match results and opponent strength. Official ranking points supply a current prior.
- **Surface-specific partial pooling.** Separate hard, clay, and grass ratings are shrunk toward overall ability in proportion to surface sample size.
- **Bayesian serve and return skill.** Observed serve- and return-point rates form beta posteriors with tour-level priors. Effective sample sizes are capped so large histories do not create false certainty.
- **Score-level Monte Carlo.** Each forecast draws latent player ability 36 times. For every draw, it simulates points, advantage games, service alternation, 6–6 tiebreaks, sets, and the selected best-of-three or best-of-five format. A standard run contains 5,040 complete matches.

The reported 80% interval is the 10th–90th percentile of win rates across posterior skill draws. It estimates parameter uncertainty inside this model; it is not a guarantee that reality will fall inside the interval.

## Custom inputs

Custom factors do not become eight additive weights. They affect different parts of the generative process:

- serve and return set beta-posterior centers;
- movement alters the opponent-adjusted return distribution;
- pressure execution changes only high-leverage points and tiebreaks;
- recent form moves a partially pooled short-term latent rating;
- fitness creates set-by-set fatigue drift;
- surface comfort creates a partially pooled court-specific rating; and
- experience reduces posterior uncertainty.

## Data

- Official ATP/WTA rankings snapshot: 2026-08-18.
- Historical result and point-stat snapshot: through 2026-05-25.
- Historical data compiler: Jeff Sackmann / Tennis Abstract.
- License for historical and derived player aggregates: CC BY-NC-SA 4.0. See `web/data/NOTICE.md`.

Rankings are deliberately newer than the match-history snapshot. The interface discloses both dates rather than implying live point statistics.

## Known limitations

- No automatic injury, travel, altitude, weather, draw, coaching, or same-day news ingestion.
- Ranking and historical snapshots can drift from current reality until the data-generation script is run and reviewed.
- Surface Elo and beta priors are research choices that require time-split backtesting and calibration before accuracy claims.
- Custom 1–10 inputs are subjective and intentionally receive wider posterior uncertainty.
- Historical coverage and stat availability vary by player, event, and tour.

## Evaluation roadmap

The next model milestone is a rolling-origin backtest that records log loss, Brier score, calibration curves, and accuracy against ranking-only, overall-Elo, and surface-Elo baselines. Results should be versioned by data cutoff and model commit.

# Baseline Labs model card

## Intended use

Baseline Labs is research software for exploring tennis forecasts. It supports four workflows:

1. current and historical ATP/WTA profiles derived from ranking snapshots and match results;
2. open-ended custom profiles expressed as direct statistical inputs;
3. user-authored single-elimination tournament brackets; and
4. current tournament draws whose completed results and unresolved matchups are refreshed from a live scoreboard.

It is not validated for wagering, player health decisions, selection decisions, or any other high-stakes use.

## Architecture

The browser model has five layers:

- **Dynamic paired-comparison ability.** Sequential Elo-style ratings learn from match results and opponent strength. Official ranking points supply a current prior.
- **Surface-specific partial pooling.** Separate hard, clay, and grass ratings are shrunk toward overall ability in proportion to surface sample size.
- **Bayesian serve and return skill.** Observed serve- and return-point rates form beta posteriors with tour-level priors. Effective sample sizes are capped so large histories do not create false certainty.
- **Score-level Monte Carlo.** Each forecast draws latent player ability 36 times. For every draw, it simulates points, advantage games, service alternation, 6–6 tiebreaks, sets, and the selected best-of-three or best-of-five format. A standard run contains 5,040 complete matches.
- **Bracket propagation.** Custom and live tournament paths preserve actual completed results and byes, then propagate forecast winners through each unresolved single-elimination round. Large custom fields use fewer posterior draws per individual match to keep browser execution responsive.

The reported 80% interval is the 10th–90th percentile of win rates across posterior skill draws. It estimates parameter uncertainty inside this model; it is not a guarantee that reality will fall inside the interval.

## Custom inputs

Custom inputs have no fixed 1–10 UI scale. They provide raw rating, uncertainty, serve/return rates, form, sample size, and open-ended pressure/endurance indices. They affect different parts of the generative process:

- serve and return set beta-posterior centers;
- pressure execution changes only high-leverage points and tiebreaks;
- recent form supplies a partially pooled short-term result signal;
- fitness creates set-by-set fatigue drift;
- a direct surface rating controls court-specific latent ability; and
- evidence sample and rating uncertainty control posterior width.

Extreme inputs are accepted by the interface, then converted through stable probability bounds or smooth transforms where an unbounded value would otherwise make simulation invalid.

## Data

- Official ATP/WTA rankings snapshot: 2026-08-18.
- Historical result and point-stat snapshot: through 2026-05-25.
- Searchable historical catalogue: 7,255 profiles (3,513 ATP and 3,742 WTA), 1967–2026, minimum five top-level recorded matches.
- Historical data compiler: Jeff Sackmann / Tennis Abstract.
- License for historical and derived player aggregates: CC BY-NC-SA 4.0. See `web/data/NOTICE.md`.
- Live draw layer: ESPN tennis scoreboard, requested on demand and cached briefly. Live timing can trail official tournament sources.

Rankings are deliberately newer than the match-history snapshot. The interface discloses both dates rather than implying live point statistics.

## Known limitations

- No automatic injury, travel, altitude, weather, coaching, or same-day news ingestion.
- Ranking and historical snapshots can drift from current reality until the data-generation script is run and reviewed.
- Surface Elo and beta priors are research choices that require time-split backtesting and calibration before accuracy claims.
- Open-ended custom inputs are subjective and can create unrealistic counterfactuals.
- Retired-player profiles represent estimated career-peak ability, not an age-specific season unless the data model is extended.
- Live forecasts are pre-match estimates. They do not update point by point from an in-progress score.
- Historical coverage and stat availability vary by player, event, and tour.

## Evaluation roadmap

The next model milestone is a rolling-origin backtest that records log loss, Brier score, calibration curves, and accuracy against ranking-only, overall-Elo, and surface-Elo baselines. Results should be versioned by data cutoff and model commit.

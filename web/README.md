# Baseline Labs web studio

The primary Baseline Labs product is a vinext/React application deployable through OpenAI Sites. The 1v1 and custom-bracket engines run in the browser. Historical profile search uses a read-only D1 catalogue, and the live tour API reads short-lived ATP/WTA scoreboard data without storing user input.

## Commands

```bash
pnpm install --frozen-lockfile --ignore-scripts
pnpm dev
pnpm exec tsc --noEmit
pnpm lint
pnpm test:model
pnpm build
node --test tests/rendered-html.test.mjs
```

## Important files

- `app/PredictionStudio.tsx` — 1v1, custom bracket, and live-tour workspaces
- `app/PlayerSearch.tsx` — debounced any-era catalogue search
- `app/api/players/route.ts` — indexed D1 player lookup
- `app/api/live/route.ts` — current tournament draw and forecast layer
- `lib/model.ts` — posterior sampling and tennis scoring simulation
- `lib/bracket.ts` — single-elimination bracket propagation and bye handling
- `lib/player-database.ts` — prepared D1 queries and model-profile conversion
- `data/players.generated.ts` — versioned derived professional profiles
- `db/schema.ts` and `drizzle/` — historical catalogue schema and migrations
- `data/NOTICE.md` — historical-data attribution and license notice
- `.openai/hosting.json` — Sites project bindings

See the repository-level `MODEL_CARD.md` for the statistical design and limitations.

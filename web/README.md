# Baseline Labs web studio

The primary Baseline Labs product is a vinext/React application deployable through OpenAI Sites. It performs all forecasts locally in the browser; no user-entered player profile is uploaded or persisted.

## Commands

```bash
pnpm install --frozen-lockfile
pnpm dev
pnpm exec tsc --noEmit
pnpm lint
pnpm test:model
pnpm build
node --test tests/rendered-html.test.mjs
```

## Important files

- `app/PredictionStudio.tsx` — professional and custom workflows plus model output
- `lib/model.ts` — posterior sampling and tennis scoring simulation
- `lib/profile-guidance.ts` — custom-factor anchors and dynamic interpretations
- `data/players.generated.ts` — versioned derived professional profiles
- `data/NOTICE.md` — historical-data attribution and license notice
- `.openai/hosting.json` — Sites project bindings

See the repository-level `MODEL_CARD.md` for the statistical design and limitations.

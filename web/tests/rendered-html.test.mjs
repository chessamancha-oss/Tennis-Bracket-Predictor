import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);
  return worker.fetch(
    new Request("http://localhost/", { headers: { accept: "text/html" } }),
    { ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) } },
    { waitUntil() {}, passThroughOnException() {} },
  );
}

test("server-renders the complete forecasting product shell", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);
  const html = await response.text();
  assert.match(html, /<title>Baseline Labs — Serious Tennis Forecasting<\/title>/i);
  assert.match(html, /Every match has a hidden shape/);
  assert.match(html, /Professional library/);
  assert.match(html, /Unrestricted profile/);
  assert.match(html, /Tournament builder/);
  assert.match(html, /Live tour desk/);
  assert.match(html, /og-baseline-labs-v2\.png/);
  assert.match(html, /LIVE CONTEXT INTELLIGENCE/);
  assert.match(html, /injury reporting/);
  assert.match(html, /Serious mechanics/);
  assert.match(html, /7,255/);
  assert.match(html, /5,040/);
  assert.doesNotMatch(html, /Your site is taking shape|react-loading-skeleton|codex-preview/i);
});

test("ships the scoring model, historical catalogue, bracket engine, live feed, and context layer", async () => {
  const [model, players, summary, bracket, studio, liveRoute, ledger, contextRoute, contextData, migration, scorecardMigration] = await Promise.all([
    readFile(new URL("../lib/model.ts", import.meta.url), "utf8"),
    readFile(new URL("../data/players.generated.ts", import.meta.url), "utf8"),
    readFile(new URL("../data/player-database-summary.generated.ts", import.meta.url), "utf8"),
    readFile(new URL("../lib/bracket.ts", import.meta.url), "utf8"),
    readFile(new URL("../app/PredictionStudio.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/api/live/route.ts", import.meta.url), "utf8"),
    readFile(new URL("../lib/prediction-ledger.ts", import.meta.url), "utf8"),
    readFile(new URL("../app/api/context/route.ts", import.meta.url), "utf8"),
    readFile(new URL("../lib/context-data.ts", import.meta.url), "utf8"),
    readFile(new URL("../drizzle/0000_late_zarek.sql", import.meta.url), "utf8"),
    readFile(new URL("../drizzle/0002_lucky_tinkerer.sql", import.meta.url), "utf8"),
  ]);
  assert.match(model, /function beta\(/);
  assert.match(model, /function simulateGame\(/);
  assert.match(model, /function simulateTieBreak\(/);
  assert.match(model, /requestedPosteriorDraws = 36/);
  assert.match(model, /advancedProfile/);
  assert.equal((players.match(/"rankingSnapshot": "2026-08-18"/g) ?? []).length, 32);
  assert.equal((players.match(/"historyCutoff": "2026-05-25"/g) ?? []).length, 32);
  assert.match(summary, /"count": 7255/);
  assert.match(summary, /"firstYear": 1967/);
  assert.match(bracket, /automatic|bye/i);
  assert.match(studio, /No 1–10 ceiling/);
  assert.match(studio, /No fixed participant cap/);
  assert.match(liveRoute, /site\.api\.espn\.com/);
  assert.match(liveRoute, /predictMatch/);
  assert.match(liveRoute, /recordAndGradeLivePredictions/);
  assert.match(ledger, /INSERT OR IGNORE INTO live_predictions/);
  assert.match(ledger, /actual_winner IS NULL/);
  assert.match(contextRoute, /buildMatchContext/);
  assert.match(contextRoute, /applyContextAdjustment/);
  assert.match(contextData, /api\.open-meteo\.com/);
  assert.match(contextData, /news\.google\.com/);
  assert.match(contextData, /recentTravel/);
  assert.match(migration, /idx_players_search_key/);
  assert.match(migration, /idx_players_tour_era/);
  assert.match(scorecardMigration, /live_predictions/);
});

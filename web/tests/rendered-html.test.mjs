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
  assert.match(html, /<title>Baseline Labs — Probabilistic Tennis Forecasts<\/title>/i);
  assert.match(html, /See the match/);
  assert.match(html, /Professional players/);
  assert.match(html, /Build your own/);
  assert.match(html, /Mechanics before mystique/);
  assert.match(html, /5,040/);
  assert.doesNotMatch(html, /Your site is taking shape|react-loading-skeleton|codex-preview/i);
});

test("ships a probabilistic scoring model and 32 versioned tour profiles", async () => {
  const [model, players, guidance] = await Promise.all([
    readFile(new URL("../lib/model.ts", import.meta.url), "utf8"),
    readFile(new URL("../data/players.generated.ts", import.meta.url), "utf8"),
    readFile(new URL("../lib/profile-guidance.ts", import.meta.url), "utf8"),
  ]);
  assert.match(model, /function beta\(/);
  assert.match(model, /function simulateGame\(/);
  assert.match(model, /function simulateTieBreak\(/);
  assert.match(model, /posteriorDraws = 36/);
  assert.equal((players.match(/"rankingSnapshot": "2026-08-18"/g) ?? []).length, 32);
  assert.equal((players.match(/"historyCutoff": "2026-05-25"/g) ?? []).length, 32);
  assert.match(guidance, /Pressure execution|beta posterior/);
});

import assert from "node:assert/strict";
import test from "node:test";
import { professionalPlayers } from "../data/players.generated";
import { customProfile, predictMatch, type PlayerProfile } from "../lib/model";
import { defaultRatings, interpretation } from "../lib/profile-guidance";

const profiles = professionalPlayers.map((player) => ({
  ...player,
  surfaceRating: { ...player.surfaceRating },
  surfaceSamples: { ...player.surfaceSamples },
})) as PlayerProfile[];

test("posterior match simulation is deterministic and internally coherent", () => {
  const first = predictMatch(profiles[0], profiles[1], "hard", 3);
  const second = predictMatch(profiles[0], profiles[1], "hard", 3);
  assert.deepEqual(first, second);
  assert.equal(first.simulations, 5040);
  assert.equal(first.posteriorDraws, 36);
  assert.ok(Math.abs(first.playerOneProbability + first.playerTwoProbability - 1) < 1e-12);
  assert.ok(first.intervalLow >= 0 && first.intervalHigh <= 1);
  assert.ok(first.intervalLow <= first.intervalHigh);
  assert.ok(first.expectedSets >= 2 && first.expectedSets <= 3);
  assert.ok(first.expectedGames > 12);
  assert.match(first.likelySetScore, /^2–[01]$/);
});

test("best-of-five simulation honors the longer match format", () => {
  const result = predictMatch(profiles[2], profiles[3], "clay", 5, 2160);
  assert.equal(result.simulations, 2160);
  assert.ok(result.expectedSets >= 3 && result.expectedSets <= 5);
  assert.match(result.likelySetScore, /^3–[0-2]$/);
});

test("custom profile choices change distributions, not direct score weights", () => {
  const novice = customProfile("custom-one", "Novice", { ...defaultRatings, experience: 1, serve: 2 }, "hard");
  const veteran = customProfile("custom-two", "Veteran", { ...defaultRatings, experience: 10, serve: 10 }, "hard");
  assert.ok(veteran.servePointsWon > novice.servePointsWon);
  assert.ok(veteran.ratingSigma < novice.ratingSigma);
  assert.match(interpretation("serve", 10, "Veteran"), /10\/10/);
  assert.match(interpretation("serve", 10, "Veteran"), /outlier/i);
});

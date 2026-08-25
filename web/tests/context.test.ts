import assert from "node:assert/strict";
import test from "node:test";
import { professionalPlayers } from "../data/players.generated";
import { applyContextAdjustment, deriveContextAdjustment, type NewsSignal, type VenueConditions } from "../lib/context";

const player = professionalPlayers[0];
const travel = {
  previousEvent: "Prior event",
  previousVenue: "Tokyo, Japan",
  lastPlayedAt: "2026-08-23T12:00:00Z",
  daysRest: 1.2,
  distanceKm: 10_900,
  estimatedTimezoneShift: 9,
  confidence: "estimated" as const,
};
const conditions: VenueConditions = {
  location: "New York, United States",
  latitude: 40.71,
  longitude: -74.01,
  elevationM: 850,
  timezone: "America/New_York",
  observedAt: "2026-08-24T18:00",
  temperatureF: 96,
  apparentTemperatureF: 101,
  humidityPercent: 72,
  precipitationIn: 0.03,
  windMph: 16,
  gustMph: 25,
  weatherCode: 61,
  sourceUrl: "https://open-meteo.com/en/docs",
};

function signal(overrides: Partial<NewsSignal> = {}): NewsSignal {
  return {
    id: "signal-1",
    player: player.name,
    kind: "injury",
    severity: "material",
    direction: "adverse",
    title: `${player.name} has an injury concern`,
    source: "example.test",
    url: "https://example.test/report",
    publishedAt: "2026-08-24T16:00:00Z",
    confidence: "reported",
    ...overrides,
  };
}

test("one unverified injury headline widens uncertainty without a direct penalty", () => {
  const adjustment = deriveContextAdjustment({ player, conditions: null, news: [signal()] });
  assert.equal(adjustment.ratingDelta, 0);
  assert.ok(adjustment.uncertaintyDelta >= 20);
  assert.equal(adjustment.availability, "questionable");
});

test("independent injury reports can become a bounded directional signal", () => {
  const adjustment = deriveContextAdjustment({
    player,
    conditions: null,
    news: [signal(), signal({ id: "signal-2", source: "second.test", url: "https://second.test/report" })],
  });
  assert.equal(adjustment.ratingDelta, -22);
  assert.ok(adjustment.factors.some((factor) => factor.confidence === "corroborated"));
});

test("tight long-haul travel and difficult weather stay bounded and auditable", () => {
  const adjustment = deriveContextAdjustment({ player, conditions, travel, news: [] });
  assert.ok(adjustment.ratingDelta < 0);
  assert.ok(adjustment.ratingDelta >= -70);
  assert.ok(adjustment.uncertaintyDelta <= 70);
  assert.ok(adjustment.factors.some((factor) => factor.id === "travel"));
  assert.ok(adjustment.factors.some((factor) => factor.id === "wind"));
  assert.ok(adjustment.factors.some((factor) => factor.id === "heat"));
});

test("context changes the latent profile rather than editing the final percentage", () => {
  const adjustment = deriveContextAdjustment({ player, conditions, travel, news: [signal({ confidence: "verified", severity: "high", kind: "availability" })] });
  const adjusted = applyContextAdjustment(player, adjustment);
  assert.equal(adjustment.availability, "withdrawn");
  assert.equal(adjusted.rating, player.rating + adjustment.ratingDelta);
  assert.equal(adjusted.surfaceRating.hard, player.surfaceRating.hard + adjustment.ratingDelta);
  assert.notEqual(adjusted.servePointsWon, player.servePointsWon);
  assert.ok(adjusted.ratingSigma > player.ratingSigma);
});

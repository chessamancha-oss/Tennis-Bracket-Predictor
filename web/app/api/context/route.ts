import { NextResponse } from "next/server";
import { applyContextAdjustment } from "../../../lib/context";
import { buildMatchContext } from "../../../lib/context-data";
import { predictMatch, type Surface, type Tour } from "../../../lib/model";
import { catalogueByNames, fallbackTourProfile, normalizePlayerName } from "../../../lib/player-database";

interface ContextRequest {
  playerOne?: string;
  playerTwo?: string;
  tour?: Tour;
  surface?: Surface;
  bestOf?: 3 | 5;
  eventName?: string;
  venue?: string;
  startsAt?: string | null;
}

function safeText(value: unknown, fallback: string, limit = 100) {
  return typeof value === "string" && value.trim() ? value.trim().slice(0, limit) : fallback;
}

function availabilityForecast(baseline: ReturnType<typeof predictMatch>, playerOne: string, playerTwo: string, oneWithdrawn: boolean) {
  return {
    ...baseline,
    playerOneProbability: oneWithdrawn ? 0 : 1,
    playerTwoProbability: oneWithdrawn ? 1 : 0,
    intervalLow: oneWithdrawn ? 0 : 1,
    intervalHigh: oneWithdrawn ? 0 : 1,
    projectedWinner: oneWithdrawn ? playerTwo : playerOne,
    likelySetScore: "W/O",
    expectedSets: 0,
    expectedGames: 0,
    tieBreakChance: 0,
    confidence: "Strong" as const,
    evidence: [{
      label: "Confirmed availability",
      leader: oneWithdrawn ? "two" as const : "one" as const,
      detail: `${oneWithdrawn ? playerOne : playerTwo} has a trusted or independently corroborated withdrawal signal for the current event context`,
      strength: 1,
    }],
  };
}

export async function POST(request: Request) {
  let input: ContextRequest;
  try {
    input = await request.json() as ContextRequest;
  } catch {
    return NextResponse.json({ error: "Invalid context request." }, { status: 400 });
  }
  const playerOneName = safeText(input.playerOne, "Player one", 80);
  const playerTwoName = safeText(input.playerTwo, "Player two", 80);
  if (playerOneName === "Player one" || playerTwoName === "Player two") {
    return NextResponse.json({ error: "Two player names are required." }, { status: 400 });
  }
  const tour: Tour = input.tour === "WTA" ? "WTA" : "ATP";
  const surface: Surface = input.surface === "clay" || input.surface === "grass" ? input.surface : "hard";
  const bestOf: 3 | 5 = input.bestOf === 5 ? 5 : 3;
  const eventName = safeText(input.eventName, "Current tour event", 120);
  const venue = safeText(input.venue, "New York, United States", 120);
  const startsAt = typeof input.startsAt === "string" && !Number.isNaN(Date.parse(input.startsAt)) ? input.startsAt : null;

  try {
    const catalogue = await catalogueByNames([playerOneName, playerTwoName]);
    const one = catalogue.get(normalizePlayerName(playerOneName)) ?? fallbackTourProfile(playerOneName, tour);
    const two = catalogue.get(normalizePlayerName(playerTwoName)) ?? fallbackTourProfile(playerTwoName, tour);
    const report = await buildMatchContext({ players: [one, two], tour, surface, eventName, venue, startsAt });
    const adjustedOne = applyContextAdjustment(one, report.players[0]);
    const adjustedTwo = applyContextAdjustment(two, report.players[1]);
    const baseline = predictMatch(one, two, surface, bestOf, 2_520, 28);
    const simulatedForecast = predictMatch(adjustedOne, adjustedTwo, surface, bestOf, 2_520, 28);
    const oneWithdrawn = report.players[0].availability === "withdrawn";
    const twoWithdrawn = report.players[1].availability === "withdrawn";
    const forecast = oneWithdrawn !== twoWithdrawn
      ? availabilityForecast(simulatedForecast, one.name, two.name, oneWithdrawn)
      : simulatedForecast;
    return NextResponse.json({
      report,
      baseline,
      forecast,
      probabilityDelta: forecast.playerOneProbability - baseline.playerOneProbability,
      source: "Baseline Context Intelligence v1",
    }, {
      headers: { "Cache-Control": "private, max-age=90" },
    });
  } catch (error) {
    return NextResponse.json({ error: error instanceof Error ? error.message : "Context intelligence unavailable." }, { status: 502 });
  }
}

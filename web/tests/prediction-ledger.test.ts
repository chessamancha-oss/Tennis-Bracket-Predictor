import assert from "node:assert/strict";
import test from "node:test";
import { captureCandidates, emptyTournamentAccuracy, resultCandidates, voidCandidates, type LedgerTournament } from "../lib/prediction-ledger-core";

const observedAt = "2026-08-25T16:00:00.000Z";

function event(matches: LedgerTournament["matches"]): LedgerTournament {
  return { id: "ATP-189-2026", tour: "ATP", name: "US Open", matches };
}

test("captures only eligible pre-match forecasts and freezes winner probability", () => {
  const captures = captureCandidates([event([
    {
      id: "eligible",
      round: "Round 1",
      startsAt: "2026-08-25T18:00:00.000Z",
      state: "pre",
      status: "Scheduled",
      players: [{ name: "Player One", winner: false }, { name: "Player Two", winner: false }],
      forecast: { winner: "Player Two", firstProbability: 0.36 },
    },
    {
      id: "already-finished",
      round: "Round 1",
      startsAt: "2026-08-25T14:00:00.000Z",
      state: "post",
      status: "Final",
      players: [{ name: "Past Winner", winner: true }, { name: "Past Loser", winner: false }],
      forecast: null,
    },
    {
      id: "unresolved-draw-slot",
      round: "Round 2",
      startsAt: null,
      state: "pre",
      status: "Scheduled",
      players: [{ name: "TBD", winner: false }, { name: "Player Three", winner: false }],
      forecast: null,
    },
    {
      id: "cancelled-before-start",
      round: "Round 1",
      startsAt: "2026-08-25T19:00:00.000Z",
      state: "pre",
      status: "Cancelled",
      players: [{ name: "Player Four", winner: false }, { name: "Player Five", winner: false }],
      forecast: { winner: "Player Four", firstProbability: 0.55 },
    },
  ])], observedAt);

  assert.equal(captures.length, 1);
  assert.equal(captures[0].id, "ATP-189-2026:eligible");
  assert.equal(captures[0].predictedWinner, "Player Two");
  assert.equal(captures[0].predictedProbability, 0.64);
  assert.equal(captures[0].predictedAt, observedAt);
});

test("grades final winners without manufacturing predictions for completed matches", () => {
  const results = resultCandidates([event([
    {
      id: "final",
      round: "Round 1",
      startsAt: "2026-08-25T14:00:00.000Z",
      state: "post",
      status: "Final",
      players: [{ name: "Actual Winner", winner: true }, { name: "Actual Loser", winner: false }],
      forecast: null,
    },
    {
      id: "in-play",
      round: "Round 1",
      startsAt: "2026-08-25T15:30:00.000Z",
      state: "in",
      status: "In Progress",
      players: [{ name: "Player One", winner: false }, { name: "Player Two", winner: false }],
      forecast: null,
    },
  ])], observedAt);

  assert.deepEqual(results, [{ id: "ATP-189-2026:final", actualWinner: "Actual Winner", resolvedAt: observedAt }]);
});

test("voids walkovers instead of grading them as winner predictions", () => {
  const tournaments = [event([
    {
      id: "walkover",
      round: "Qualifying Final",
      startsAt: "2026-08-25T16:30:00.000Z",
      state: "post",
      status: "Walkover",
      players: [{ name: "Official Advancer", winner: true }, { name: "Withdrawn Player", winner: false }],
      forecast: null,
    },
  ])];

  assert.deepEqual(resultCandidates(tournaments, observedAt), []);
  assert.deepEqual(voidCandidates(tournaments, observedAt), [{
    id: "ATP-189-2026:walkover",
    reason: "Walkover",
    voidedAt: observedAt,
  }]);
});

test("empty scorecards expose accuracy and calibration metrics without inventing values", () => {
  assert.deepEqual(emptyTournamentAccuracy(), {
    captured: 0,
    pending: 0,
    graded: 0,
    correct: 0,
    wrong: 0,
    voided: 0,
    accuracy: null,
    averageConfidence: null,
    brierScore: null,
    trackingSince: null,
    lastGradedAt: null,
  });
});

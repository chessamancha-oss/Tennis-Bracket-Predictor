import assert from "node:assert/strict";
import test from "node:test";
import { captureCandidates, resultCandidates, type LedgerTournament } from "../lib/prediction-ledger-core";

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
      players: [{ name: "Player One", winner: false }, { name: "Player Two", winner: false }],
      forecast: { winner: "Player Two", firstProbability: 0.36 },
    },
    {
      id: "already-finished",
      round: "Round 1",
      startsAt: "2026-08-25T14:00:00.000Z",
      state: "post",
      players: [{ name: "Past Winner", winner: true }, { name: "Past Loser", winner: false }],
      forecast: null,
    },
    {
      id: "unresolved-draw-slot",
      round: "Round 2",
      startsAt: null,
      state: "pre",
      players: [{ name: "TBD", winner: false }, { name: "Player Three", winner: false }],
      forecast: null,
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
      players: [{ name: "Actual Winner", winner: true }, { name: "Actual Loser", winner: false }],
      forecast: null,
    },
    {
      id: "in-play",
      round: "Round 1",
      startsAt: "2026-08-25T15:30:00.000Z",
      state: "in",
      players: [{ name: "Player One", winner: false }, { name: "Player Two", winner: false }],
      forecast: null,
    },
  ])], observedAt);

  assert.deepEqual(results, [{ id: "ATP-189-2026:final", actualWinner: "Actual Winner", resolvedAt: observedAt }]);
});

import { env } from "cloudflare:workers";
import { LIVE_MODEL_VERSION, captureCandidates, resultCandidates, type LedgerTournament, type TournamentAccuracy } from "./prediction-ledger-core";

export { emptyTournamentAccuracy } from "./prediction-ledger-core";

interface AccuracyRow {
  tournament_id: string;
  captured: number;
  pending: number;
  graded: number;
  correct: number;
  tracking_since: string | null;
  last_graded_at: string | null;
}

interface DatabaseEnv {
  DB?: D1Database;
}

function database() {
  return (env as unknown as DatabaseEnv).DB;
}

function chunks<T>(items: T[], size = 75) {
  const result: T[][] = [];
  for (let index = 0; index < items.length; index += size) result.push(items.slice(index, index + size));
  return result;
}

export async function recordAndGradeLivePredictions(tournaments: LedgerTournament[], observedAt: string): Promise<Map<string, TournamentAccuracy>> {
  const db = database();
  if (!db || tournaments.length === 0) return new Map();

  for (const group of chunks(captureCandidates(tournaments, observedAt))) {
    await db.batch(group.map((capture) => db.prepare(`
      INSERT OR IGNORE INTO live_predictions (
        id, tournament_id, tour, tournament_name, round, match_id,
        player_one, player_two, predicted_winner, predicted_probability,
        predicted_at, starts_at, model_version
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `).bind(
      capture.id,
      capture.tournamentId,
      capture.tour,
      capture.tournamentName,
      capture.round,
      capture.matchId,
      capture.playerOne,
      capture.playerTwo,
      capture.predictedWinner,
      capture.predictedProbability,
      capture.predictedAt,
      capture.startsAt,
      LIVE_MODEL_VERSION,
    )));
  }

  for (const group of chunks(resultCandidates(tournaments, observedAt))) {
    await db.batch(group.map((result) => db.prepare(`
      UPDATE live_predictions
      SET actual_winner = ?,
          correct = CASE WHEN predicted_winner = ? THEN 1 ELSE 0 END,
          resolved_at = ?
      WHERE id = ? AND actual_winner IS NULL
    `).bind(result.actualWinner, result.actualWinner, result.resolvedAt, result.id)));
  }

  const tournamentIds = [...new Set(tournaments.map((tournament) => tournament.id))];
  const placeholders = tournamentIds.map(() => "?").join(", ");
  const rows = await db.prepare(`
    SELECT tournament_id,
           COUNT(*) AS captured,
           SUM(CASE WHEN actual_winner IS NULL THEN 1 ELSE 0 END) AS pending,
           SUM(CASE WHEN actual_winner IS NOT NULL THEN 1 ELSE 0 END) AS graded,
           SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END) AS correct,
           MIN(predicted_at) AS tracking_since,
           MAX(resolved_at) AS last_graded_at
    FROM live_predictions
    WHERE tournament_id IN (${placeholders})
    GROUP BY tournament_id
  `).bind(...tournamentIds).all<AccuracyRow>();

  return new Map(rows.results.map((row) => {
    const graded = Number(row.graded ?? 0);
    const correct = Number(row.correct ?? 0);
    const value: TournamentAccuracy = {
      captured: Number(row.captured ?? 0),
      pending: Number(row.pending ?? 0),
      graded,
      correct,
      wrong: Math.max(0, graded - correct),
      accuracy: graded ? correct / graded : null,
      trackingSince: row.tracking_since,
      lastGradedAt: row.last_graded_at,
    };
    return [row.tournament_id, value];
  }));
}

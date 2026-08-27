export const LIVE_MODEL_VERSION = "baseline-v3.1";

export interface LedgerMatch {
  id: string;
  round: string;
  startsAt: string | null;
  state: string;
  status: string;
  players: Array<{ name: string; winner: boolean }>;
  forecast: null | { winner: string; firstProbability: number };
}

export interface LedgerTournament {
  id: string;
  tour: string;
  name: string;
  matches: LedgerMatch[];
}

export interface TournamentAccuracy {
  captured: number;
  pending: number;
  graded: number;
  correct: number;
  wrong: number;
  voided: number;
  accuracy: number | null;
  averageConfidence: number | null;
  brierScore: number | null;
  trackingSince: string | null;
  lastGradedAt: string | null;
}

export interface PredictionCapture {
  id: string;
  tournamentId: string;
  tour: string;
  tournamentName: string;
  round: string;
  matchId: string;
  playerOne: string;
  playerTwo: string;
  predictedWinner: string;
  predictedProbability: number;
  predictedAt: string;
  startsAt: string | null;
}

export interface PredictionResult {
  id: string;
  actualWinner: string;
  resolvedAt: string;
}

export interface PredictionVoid {
  id: string;
  reason: string;
  voidedAt: string;
}

const clamp = (value: number, low: number, high: number) => Math.max(low, Math.min(high, value));

function predictionId(tournamentId: string, matchId: string) {
  return `${tournamentId}:${matchId}`;
}

export function captureCandidates(tournaments: LedgerTournament[], observedAt: string): PredictionCapture[] {
  return tournaments.flatMap((tournament) => tournament.matches.flatMap((match) => {
    const first = match.players[0]?.name?.trim();
    const second = match.players[1]?.name?.trim();
    const forecast = match.forecast;
    if (match.state !== "pre" || nonPlayedStatus(match.status) || !forecast || !first || !second || first === "TBD" || second === "TBD") return [];
    if (forecast.winner !== first && forecast.winner !== second) return [];
    const winnerProbability = forecast.winner === first ? forecast.firstProbability : 1 - forecast.firstProbability;
    return [{
      id: predictionId(tournament.id, match.id),
      tournamentId: tournament.id,
      tour: tournament.tour,
      tournamentName: tournament.name,
      round: match.round,
      matchId: match.id,
      playerOne: first,
      playerTwo: second,
      predictedWinner: forecast.winner,
      predictedProbability: clamp(winnerProbability, 0, 1),
      predictedAt: observedAt,
      startsAt: match.startsAt,
    }];
  }));
}

export function resultCandidates(tournaments: LedgerTournament[], observedAt: string): PredictionResult[] {
  return tournaments.flatMap((tournament) => tournament.matches.flatMap((match) => {
    if (match.state !== "post" || nonPlayedStatus(match.status)) return [];
    const winner = match.players.find((player) => player.winner)?.name?.trim();
    if (!winner || winner === "TBD") return [];
    return [{ id: predictionId(tournament.id, match.id), actualWinner: winner, resolvedAt: observedAt }];
  }));
}

function nonPlayedStatus(status: string) {
  return /walkover|\bw\/?o\b|\bbye\b|cancel(?:led|ed)|abandon(?:ed|ment)|postpone(?:d|ment)|no contest/i.test(status);
}

export function voidCandidates(tournaments: LedgerTournament[], observedAt: string): PredictionVoid[] {
  return tournaments.flatMap((tournament) => tournament.matches.flatMap((match) => {
    if (!nonPlayedStatus(match.status)) return [];
    return [{
      id: predictionId(tournament.id, match.id),
      reason: match.status.trim() || "Match not played",
      voidedAt: observedAt,
    }];
  }));
}

export function emptyTournamentAccuracy(): TournamentAccuracy {
  return { captured: 0, pending: 0, graded: 0, correct: 0, wrong: 0, voided: 0, accuracy: null, averageConfidence: null, brierScore: null, trackingSince: null, lastGradedAt: null };
}

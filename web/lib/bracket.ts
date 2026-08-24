import { predictMatch, type PlayerProfile, type Surface } from "./model";

export interface BracketParticipant {
  id: string;
  name: string;
  profile: PlayerProfile;
}

export interface ForecastBracketMatch {
  id: string;
  round: number;
  roundLabel: string;
  slot: number;
  one: BracketParticipant | null;
  two: BracketParticipant | null;
  winner: BracketParticipant | null;
  winnerProbability: number | null;
  firstProbability: number | null;
  score: string;
  bye: boolean;
}

export interface ForecastBracket {
  size: number;
  rounds: Array<{ label: string; matches: ForecastBracketMatch[] }>;
  champion: BracketParticipant | null;
}

function bracketSize(count: number) {
  let size = 2;
  while (size < count) size *= 2;
  return size;
}

function roundLabel(players: number, round: number) {
  const remaining = players / 2 ** round;
  if (remaining === 1) return "Final";
  if (remaining === 2) return "Semifinals";
  if (remaining === 4) return "Quarterfinals";
  return `Round of ${remaining * 2}`;
}

export function forecastBracket(participants: BracketParticipant[], surface: Surface, bestOf: 3 | 5): ForecastBracket {
  const size = bracketSize(Math.max(2, participants.length));
  const firstRoundMatches = size / 2;
  const byeCount = size - participants.length;
  let participantIndex = 0;
  let field: Array<BracketParticipant | null> = [];
  for (let match = 0; match < firstRoundMatches; match += 1) {
    field.push(participants[participantIndex++] ?? null);
    field.push(match < byeCount ? null : participants[participantIndex++] ?? null);
  }
  const rounds: ForecastBracket["rounds"] = [];
  const simulations = Math.max(180, Math.round(24_000 / size));
  const posteriorDraws = Math.max(6, Math.min(18, Math.round(simulations / 30)));

  for (let round = 1; field.length > 1; round += 1) {
    const matches: ForecastBracketMatch[] = [];
    const next: Array<BracketParticipant | null> = [];
    const label = roundLabel(size, round);
    for (let index = 0; index < field.length; index += 2) {
      const one = field[index] ?? null;
      const two = field[index + 1] ?? null;
      if (!one || !two) {
        const winner = one ?? two;
        matches.push({
          id: `r${round}-m${index / 2}`,
          round,
          roundLabel: label,
          slot: index / 2,
          one,
          two,
          winner,
          winnerProbability: winner ? 1 : null,
          firstProbability: one ? 1 : two ? 0 : null,
          score: winner ? "BYE" : "—",
          bye: true,
        });
        next.push(winner);
        continue;
      }
      const result = predictMatch(one.profile, two.profile, surface, bestOf, simulations, posteriorDraws);
      const winner = result.projectedWinner === one.name ? one : two;
      matches.push({
        id: `r${round}-m${index / 2}`,
        round,
        roundLabel: label,
        slot: index / 2,
        one,
        two,
        winner,
        winnerProbability: winner === one ? result.playerOneProbability : result.playerTwoProbability,
        firstProbability: result.playerOneProbability,
        score: result.likelySetScore,
        bye: false,
      });
      next.push(winner);
    }
    rounds.push({ label, matches });
    field = next;
  }
  return { size, rounds, champion: field[0] ?? null };
}
